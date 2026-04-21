import copy
import html
import json
from collections import defaultdict, deque
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .run_store import RunStore


VISUALIZER_SCHEMA_VERSION = "phase2"
DEFAULT_UI_HOST = "127.0.0.1"
DEFAULT_UI_PORT = 8787

_NODE_KIND_ORDER = {
    "workflow": 0,
    "attempt": 1,
    "step": 2,
    "group_step": 2,
    "reducer": 3,
    "validator": 4,
    "router": 5,
    "replan": 6,
    "clarification": 7,
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _node_title(node: dict[str, Any]) -> str:
    for key in ("label", "task", "agent_name", "node_id"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "node"


def _node_subtitle(node: dict[str, Any]) -> str:
    parts: list[str] = []
    kind = str(node.get("kind") or "").strip()
    if kind:
        parts.append(kind)
    status = str(node.get("status") or "").strip()
    if status:
        parts.append(status)
    attempt = node.get("attempt")
    if isinstance(attempt, int):
        parts.append(f"attempt {attempt}")
    step_id = node.get("step_id")
    if isinstance(step_id, str) and step_id.strip():
        parts.append(step_id.strip())
    return " | ".join(parts)


def _node_order_key(node: dict[str, Any]) -> tuple[Any, ...]:
    node_id = str(node.get("node_id") or "")
    label = _node_title(node).lower()
    return (
        _NODE_KIND_ORDER.get(str(node.get("kind") or "").strip(), 99),
        _safe_int(node.get("attempt"), 0),
        str(node.get("step_id") or ""),
        label,
        node_id,
    )


def _graph_depths(graph: dict[str, Any]) -> dict[str, int]:
    nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
    edges = [item for item in graph.get("edges", []) if isinstance(item, dict)]
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {str(node.get("node_id") or ""): 0 for node in nodes}
    root_node_id = str(graph.get("root_node_id") or "")

    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if not source or not target:
            continue
        adjacency[source].append(target)
        indegree[target] = indegree.get(target, 0) + 1
        indegree.setdefault(source, indegree.get(source, 0))

    seeds = []
    if root_node_id:
        seeds.append(root_node_id)
    seeds.extend(
        node_id
        for node_id, degree in indegree.items()
        if degree == 0 and node_id and node_id not in seeds
    )
    if not seeds:
        seeds = [str(node.get("node_id") or "") for node in nodes if node.get("node_id")]

    depths: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds if seed)
    while queue:
        node_id, depth = queue.popleft()
        if node_id in depths and depths[node_id] <= depth:
            continue
        depths[node_id] = depth
        for target in adjacency.get(node_id, []):
            queue.append((target, depth + 1))

    max_depth = max(depths.values(), default=0)
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        if node_id and node_id not in depths:
            max_depth += 1
            depths[node_id] = max_depth
    return depths


def build_graph_view_model(graph: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(graph, dict):
        return {
            "schema_version": VISUALIZER_SCHEMA_VERSION,
            "kind": "graph_view",
            "width": 960,
            "height": 320,
            "nodes": [],
            "edges": [],
            "statistics": {},
        }

    nodes = [copy.deepcopy(item) for item in graph.get("nodes", []) if isinstance(item, dict)]
    edges = [copy.deepcopy(item) for item in graph.get("edges", []) if isinstance(item, dict)]
    depths = _graph_depths(graph)
    columns: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        columns[depths.get(node_id, 0)].append(node)

    for depth_nodes in columns.values():
        depth_nodes.sort(key=_node_order_key)

    node_width = 228
    node_height = 86
    margin_x = 64
    margin_y = 56
    column_gap = 264
    row_gap = 118
    positions: dict[str, dict[str, int]] = {}
    max_column_size = 0
    max_depth = 0
    for depth, depth_nodes in columns.items():
        max_depth = max(max_depth, depth)
        max_column_size = max(max_column_size, len(depth_nodes))
        for index, node in enumerate(depth_nodes):
            node_id = str(node.get("node_id") or "")
            positions[node_id] = {
                "x": margin_x + (depth * column_gap),
                "y": margin_y + (index * row_gap),
            }

    view_nodes = []
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        position = positions.get(node_id, {"x": margin_x, "y": margin_y})
        view_nodes.append(
            {
                "node_id": node_id,
                "kind": str(node.get("kind") or "node"),
                "status": str(node.get("status") or ""),
                "title": _node_title(node),
                "subtitle": _node_subtitle(node),
                "x": position["x"],
                "y": position["y"],
                "width": node_width,
                "height": node_height,
                "depth": depths.get(node_id, 0),
                "data": node,
            }
        )

    view_edges = []
    for edge in edges:
        source_id = str(edge.get("source") or "")
        target_id = str(edge.get("target") or "")
        source = positions.get(source_id)
        target = positions.get(target_id)
        if not source or not target:
            continue
        start_x = source["x"] + node_width
        start_y = source["y"] + (node_height // 2)
        end_x = target["x"]
        end_y = target["y"] + (node_height // 2)
        bend = max(64, (end_x - start_x) // 2)
        path = (
            f"M {start_x} {start_y} "
            f"C {start_x + bend} {start_y}, {end_x - bend} {end_y}, {end_x} {end_y}"
        )
        view_edges.append(
            {
                "edge_id": str(edge.get("edge_id") or f"{source_id}->{target_id}"),
                "source": source_id,
                "target": target_id,
                "relation": str(edge.get("relation") or ""),
                "path": path,
                "label_x": int((start_x + end_x) / 2),
                "label_y": int((start_y + end_y) / 2) - 10,
                "data": edge,
            }
        )

    width = margin_x + ((max_depth + 1) * column_gap) + node_width
    height = margin_y + (max(max_column_size, 1) * row_gap) + 120
    return {
        "schema_version": VISUALIZER_SCHEMA_VERSION,
        "kind": "graph_view",
        "width": max(width, 960),
        "height": max(height, 320),
        "nodes": view_nodes,
        "edges": view_edges,
        "statistics": copy.deepcopy(graph.get("statistics")) if isinstance(graph.get("statistics"), dict) else {},
    }


def build_graph_index(graph: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(graph, dict):
        return {
            "schema_version": VISUALIZER_SCHEMA_VERSION,
            "kind": "graph_index",
            "root_node_id": "",
            "nodes": {},
            "incoming": {},
            "outgoing": {},
            "counts": {"kinds": {}, "statuses": {}, "agents": {}},
        }

    nodes = [copy.deepcopy(item) for item in graph.get("nodes", []) if isinstance(item, dict)]
    edges = [copy.deepcopy(item) for item in graph.get("edges", []) if isinstance(item, dict)]
    depths = _graph_depths(graph)
    indexed_nodes: dict[str, dict[str, Any]] = {}
    incoming: dict[str, list[dict[str, str]]] = defaultdict(list)
    outgoing: dict[str, list[dict[str, str]]] = defaultdict(list)
    kind_counts: dict[str, int] = defaultdict(int)
    status_counts: dict[str, int] = defaultdict(int)
    agent_counts: dict[str, int] = defaultdict(int)

    for node in nodes:
        node_id = str(node.get("node_id") or "").strip()
        if not node_id:
            continue
        kind = str(node.get("kind") or "node").strip() or "node"
        status = str(node.get("status") or "unknown").strip() or "unknown"
        agent_name = str(node.get("agent_name") or node.get("target_agent") or "").strip()
        indexed_nodes[node_id] = {
            "node_id": node_id,
            "kind": kind,
            "status": status,
            "title": _node_title(node),
            "subtitle": _node_subtitle(node),
            "attempt": node.get("attempt"),
            "step_id": str(node.get("step_id") or "").strip(),
            "agent_name": agent_name,
            "task": str(node.get("task") or "").strip(),
            "depth": depths.get(node_id, 0),
        }
        kind_counts[kind] += 1
        status_counts[status] += 1
        if agent_name:
            agent_counts[agent_name] += 1

    for edge in edges:
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        relation = str(edge.get("relation") or "").strip()
        if not source or not target:
            continue
        outgoing[source].append({"node_id": target, "relation": relation})
        incoming[target].append({"node_id": source, "relation": relation})

    def _sorted_adjacency(values: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
        normalized: dict[str, list[dict[str, str]]] = {}
        for node_id, items in values.items():
            normalized[node_id] = sorted(
                items,
                key=lambda item: (str(item.get("relation") or ""), str(item.get("node_id") or "")),
            )
        return normalized

    return {
        "schema_version": VISUALIZER_SCHEMA_VERSION,
        "kind": "graph_index",
        "root_node_id": str(graph.get("root_node_id") or ""),
        "nodes": indexed_nodes,
        "incoming": _sorted_adjacency(incoming),
        "outgoing": _sorted_adjacency(outgoing),
        "counts": {
            "kinds": dict(sorted(kind_counts.items())),
            "statuses": dict(sorted(status_counts.items())),
            "agents": dict(sorted(agent_counts.items())),
        },
    }


def build_run_visualization_payload(inspection: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(inspection, dict):
        raise ValueError("Run inspection payload must be a dict.")
    graph = inspection.get("graph") if isinstance(inspection.get("graph"), dict) else {}
    return {
        "schema_version": VISUALIZER_SCHEMA_VERSION,
        "run_id": str(inspection.get("run_id") or ""),
        "summary": copy.deepcopy(inspection.get("summary")) if isinstance(inspection.get("summary"), dict) else {},
        "state": copy.deepcopy(inspection.get("state")) if isinstance(inspection.get("state"), dict) else {},
        "observability": copy.deepcopy(inspection.get("observability"))
        if isinstance(inspection.get("observability"), dict)
        else {},
        "graph": copy.deepcopy(graph),
        "graph_view": build_graph_view_model(graph),
        "graph_index": build_graph_index(graph),
        "graph_mermaid": str(inspection.get("graph_mermaid") or ""),
        "timeline": copy.deepcopy(inspection.get("timeline")) if isinstance(inspection.get("timeline"), list) else [],
    }


def list_run_visualizations(
    run_store: RunStore,
    *,
    limit: int = 50,
    status: str | None = None,
    task_contains: str | None = None,
    agent: str | None = None,
    has_errors: bool | None = None,
    min_duration_ms: float | None = None,
    max_duration_ms: float | None = None,
    slow_step_ms: float | None = None,
) -> dict[str, Any]:
    runs = run_store.list_runs(
        limit=limit,
        status=status,
        task_contains=task_contains,
        agent=agent,
        has_errors=has_errors,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        slow_step_ms=slow_step_ms,
    )
    return {
        "schema_version": VISUALIZER_SCHEMA_VERSION,
        "count": len(runs),
        "runs": runs,
        "base_dir": str(run_store.base_dir),
    }


def load_run_visualization(run_store: RunStore, run_id: str) -> dict[str, Any]:
    inspection = run_store.inspect(run_id, include_timeline=True)
    if not isinstance(inspection, dict):
        raise KeyError(run_id)
    payload = build_run_visualization_payload(inspection)
    payload["base_dir"] = str(run_store.base_dir)
    return payload


def load_run_graph_payload(run_store: RunStore, run_id: str, *, format: str = "view") -> dict[str, Any] | str:
    visualization = load_run_visualization(run_store, run_id)
    target_format = str(format or "view").strip().lower()
    if target_format == "mermaid":
        return visualization["graph_mermaid"]
    if target_format == "json":
        return visualization["graph"]
    if target_format == "view":
        return visualization["graph_view"]
    raise ValueError(f"Unsupported graph format '{format}'.")


def load_run_observability_payload(run_store: RunStore, run_id: str) -> dict[str, Any]:
    visualization = load_run_visualization(run_store, run_id)
    return copy.deepcopy(visualization.get("observability") or {})


def render_run_visualizer_html(*, base_dir: str = "") -> str:
    safe_base_dir = html.escape(base_dir)
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenFabric Run Visualizer</title>
  <style>
    :root {{
      --bg: #f5efe4;
      --panel: rgba(255, 249, 240, 0.82);
      --panel-strong: rgba(255, 252, 247, 0.95);
      --ink: #1b1a17;
      --muted: #6d665d;
      --line: rgba(76, 62, 46, 0.18);
      --accent: #0f766e;
      --accent-2: #b45309;
      --shadow: 0 24px 60px rgba(43, 30, 18, 0.14);
      --workflow: #e0f2fe;
      --attempt: #fef3c7;
      --step: #dcfce7;
      --validator: #fee2e2;
      --reducer: #cffafe;
      --router: #f3e8ff;
      --replan: #fde68a;
      --clarification: #fecaca;
      --default: #e7e5e4;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(180, 83, 9, 0.16), transparent 26%),
        linear-gradient(180deg, #f7f2e9 0%, #eee3d2 100%);
      min-height: 100vh;
    }}

    .shell {{
      padding: 24px;
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 20px;
      min-height: 100vh;
    }}

    .sidebar,
    .panel {{
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.45);
      border-radius: 26px;
      box-shadow: var(--shadow);
    }}

    .sidebar {{
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}

    .sidebar-head {{
      padding: 24px 24px 18px;
      border-bottom: 1px solid var(--line);
      background:
        linear-gradient(135deg, rgba(15, 118, 110, 0.1), transparent 52%),
        linear-gradient(215deg, rgba(180, 83, 9, 0.12), transparent 48%);
    }}

    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}

    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1;
      font-weight: 700;
      font-family: "IBM Plex Sans Condensed", "Avenir Next Condensed", sans-serif;
    }}

    .base-dir {{
      margin-top: 12px;
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }}

    .controls {{
      padding: 18px 20px;
      display: grid;
      gap: 10px;
      border-bottom: 1px solid var(--line);
    }}

    .controls input,
    .controls select {{
      width: 100%;
      border: 1px solid rgba(48, 43, 37, 0.12);
      border-radius: 14px;
      padding: 11px 12px;
      font: inherit;
      background: rgba(255, 255, 255, 0.75);
      color: var(--ink);
    }}

    .run-list {{
      overflow: auto;
      padding: 10px;
      display: grid;
      gap: 10px;
    }}

    .run-card {{
      border: 1px solid transparent;
      border-radius: 18px;
      padding: 14px;
      background: var(--panel-strong);
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
    }}

    .run-card:hover {{
      transform: translateY(-1px);
      border-color: rgba(15, 118, 110, 0.25);
      box-shadow: 0 12px 24px rgba(24, 18, 10, 0.08);
    }}

    .run-card.active {{
      border-color: rgba(15, 118, 110, 0.48);
      box-shadow: 0 16px 30px rgba(15, 118, 110, 0.12);
    }}

    .run-id {{
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }}

    .run-task {{
      margin-top: 7px;
      font-size: 15px;
      font-weight: 600;
      line-height: 1.3;
    }}

    .run-meta {{
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}

    .chip {{
      display: inline-flex;
      align-items: center;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 11px;
      background: rgba(27, 26, 23, 0.06);
      color: var(--muted);
    }}

    .main {{
      display: grid;
      gap: 20px;
      min-width: 0;
    }}

    .hero {{
      padding: 24px 28px;
      position: relative;
      overflow: hidden;
    }}

    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -40px -60px auto;
      width: 220px;
      height: 220px;
      background: radial-gradient(circle, rgba(180, 83, 9, 0.14), transparent 70%);
      pointer-events: none;
    }}

    .hero-title {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
    }}

    .hero h2 {{
      margin: 0;
      font-size: 30px;
      line-height: 1.05;
      font-family: "IBM Plex Serif", "Georgia", serif;
    }}

    .hero p {{
      margin: 10px 0 0;
      max-width: 70ch;
      color: var(--muted);
    }}

    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 20px;
    }}

    .summary-card {{
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(60, 43, 24, 0.08);
      border-radius: 18px;
      padding: 16px;
      min-height: 106px;
    }}

    .summary-card .label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}

    .summary-card .value {{
      margin-top: 8px;
      font-size: 24px;
      font-weight: 700;
      line-height: 1.1;
    }}

    .summary-card .detail {{
      margin-top: 8px;
      font-size: 13px;
      color: var(--muted);
      line-height: 1.35;
    }}

    .content-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.55fr) minmax(320px, 0.95fr);
      gap: 20px;
      min-width: 0;
    }}

    .graph-panel {{
      padding: 18px;
      overflow: hidden;
    }}

    .panel-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 14px;
    }}

    .panel-title {{
      font-size: 18px;
      font-weight: 700;
      margin: 0;
    }}

    .panel-subtle {{
      font-size: 12px;
      color: var(--muted);
    }}

    .graph-shell {{
      border-radius: 20px;
      overflow: auto;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(246, 238, 226, 0.82)),
        linear-gradient(90deg, rgba(15, 118, 110, 0.05), transparent 20%);
      border: 1px solid rgba(61, 46, 29, 0.09);
      min-height: 420px;
    }}

    .graph-toolbar {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) repeat(2, minmax(140px, 0.55fr)) auto auto;
      gap: 10px;
      margin-bottom: 12px;
    }}

    .graph-toolbar input,
    .graph-toolbar select {{
      width: 100%;
      border: 1px solid rgba(48, 43, 37, 0.12);
      border-radius: 14px;
      padding: 10px 12px;
      font: inherit;
      background: rgba(255, 255, 255, 0.78);
      color: var(--ink);
    }}

    .toggle-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 0 12px;
      min-height: 42px;
      border-radius: 14px;
      border: 1px solid rgba(48, 43, 37, 0.12);
      background: rgba(255, 255, 255, 0.72);
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      white-space: nowrap;
    }}

    .toggle-chip input {{
      margin: 0;
      accent-color: var(--accent);
    }}

    .legend-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }}

    .legend-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 11px;
      border-radius: 999px;
      font-size: 11px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(60, 43, 24, 0.08);
    }}

    .legend-swatch {{
      width: 11px;
      height: 11px;
      border-radius: 999px;
      display: inline-block;
      border: 1px solid rgba(60, 43, 24, 0.12);
    }}

    svg {{
      display: block;
      min-width: 100%;
    }}

    .edge {{
      fill: none;
      stroke: rgba(85, 70, 51, 0.35);
      stroke-width: 2;
    }}

    .edge-label {{
      font-size: 11px;
      fill: #7c6550;
      text-anchor: middle;
      font-family: "IBM Plex Sans", sans-serif;
    }}

    .node-card rect {{
      stroke-width: 1.4;
      filter: drop-shadow(0 10px 16px rgba(43, 31, 18, 0.12));
    }}

    .node-card text {{
      pointer-events: none;
    }}

    .node-title {{
      font-size: 13px;
      font-weight: 700;
      fill: #201d18;
    }}

    .node-subtitle {{
      font-size: 11px;
      fill: #675d55;
    }}

    .node-card.active rect {{
      stroke: #0f766e;
      stroke-width: 2.6;
    }}

    .node-card.related rect {{
      stroke: rgba(15, 118, 110, 0.78);
      stroke-width: 2.1;
      stroke-dasharray: 5 4;
    }}

    .detail-stack {{
      display: grid;
      gap: 20px;
      min-width: 0;
    }}

    .detail-panel {{
      padding: 18px;
      overflow: hidden;
    }}

    .codeblock {{
      margin: 0;
      padding: 14px;
      border-radius: 16px;
      background: rgba(24, 17, 8, 0.92);
      color: #f8eee1;
      overflow: auto;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
    }}

    .signal-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}

    .signal-card {{
      border-radius: 16px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(60, 43, 24, 0.08);
    }}

    .signal-card .label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}

    .signal-card .value {{
      margin-top: 6px;
      font-size: 20px;
      font-weight: 700;
      line-height: 1.1;
    }}

    .signal-card .detail {{
      margin-top: 6px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }}

    .adjacency-shell {{
      display: grid;
      gap: 12px;
      margin-bottom: 14px;
    }}

    .adjacency-group {{
      display: grid;
      gap: 8px;
    }}

    .adjacency-title {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}

    .adjacency-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
    }}

    .adjacency-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.76);
      border: 1px solid rgba(60, 43, 24, 0.08);
      font-size: 11px;
      color: var(--muted);
    }}

    .adjacency-pill strong {{
      color: var(--ink);
      font-weight: 700;
    }}

    .metrics-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}

    .metrics-table th,
    .metrics-table td {{
      text-align: left;
      padding: 9px 8px;
      border-bottom: 1px solid rgba(58, 44, 27, 0.08);
      vertical-align: top;
    }}

    .metrics-table th {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}

    .failure-list {{
      display: grid;
      gap: 10px;
    }}

    .failure-card {{
      border-radius: 16px;
      padding: 13px 14px;
      background: rgba(255, 250, 248, 0.9);
      border: 1px solid rgba(220, 38, 38, 0.12);
    }}

    .failure-card .title {{
      font-size: 13px;
      font-weight: 700;
    }}

    .failure-card .meta {{
      margin-top: 6px;
      font-size: 11px;
      color: var(--muted);
    }}

    .failure-card .reason {{
      margin-top: 8px;
      font-size: 12px;
      line-height: 1.45;
      color: var(--ink);
    }}

    .timeline {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}

    .timeline th,
    .timeline td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid rgba(58, 44, 27, 0.08);
      vertical-align: top;
    }}

    .timeline th {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}

    .empty,
    .loading {{
      padding: 26px;
      color: var(--muted);
      text-align: center;
    }}

    @media (max-width: 1180px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}

      .content-grid {{
        grid-template-columns: 1fr;
      }}

      .summary-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}

      .graph-toolbar {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 720px) {{
      .shell {{
        padding: 14px;
        gap: 14px;
      }}

      .summary-grid {{
        grid-template-columns: 1fr;
      }}

      .signal-grid {{
        grid-template-columns: 1fr;
      }}

      .hero h2 {{
        font-size: 24px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="sidebar-head">
        <div class="eyebrow">OpenFabric</div>
        <h1>Run Atlas</h1>
        <div class="base-dir">Store: __BASE_DIR__</div>
      </div>
      <div class="controls">
        <input id="run-search" type="search" placeholder="Search task, run id, status" />
        <select id="run-status">
          <option value="">All statuses</option>
          <option value="completed">completed</option>
          <option value="running">running</option>
          <option value="failed">failed</option>
          <option value="needs_clarification">needs_clarification</option>
        </select>
        <input id="run-agent" type="search" placeholder="Filter by agent name" />
        <select id="run-errors">
          <option value="">Any error state</option>
          <option value="true">Errors only</option>
        </select>
      </div>
      <div id="run-list" class="run-list">
        <div class="loading">Loading persisted runs...</div>
      </div>
    </aside>

    <main class="main">
      <section class="panel hero">
        <div class="hero-title">
          <div>
            <div class="eyebrow">Execution Graph</div>
            <h2 id="hero-title">Select a persisted run</h2>
            <p id="hero-subtitle">The dashboard renders workflow state, timeline checkpoints, and the execution graph from the persisted run store.</p>
          </div>
          <div id="hero-status" class="chip">idle</div>
        </div>
        <div id="summary-grid" class="summary-grid"></div>
      </section>

      <section class="content-grid">
        <section class="panel graph-panel">
          <div class="panel-head">
            <div>
              <h3 class="panel-title">Graph View</h3>
              <div class="panel-subtle">Interactive SVG view of the persisted workflow graph.</div>
            </div>
            <div id="graph-stats" class="chip">0 nodes</div>
          </div>
          <div class="graph-toolbar">
            <input id="node-search" type="search" placeholder="Search nodes, tasks, agents" />
            <select id="node-kind-filter">
              <option value="">All node kinds</option>
            </select>
            <select id="node-status-filter">
              <option value="">All node statuses</option>
            </select>
            <label class="toggle-chip" for="focus-neighborhood">
              <input id="focus-neighborhood" type="checkbox" />
              Focus selection
            </label>
            <label class="toggle-chip" for="auto-refresh">
              <input id="auto-refresh" type="checkbox" />
              Auto-refresh
            </label>
          </div>
          <div id="graph-legend" class="legend-grid"></div>
          <div id="graph-shell" class="graph-shell">
            <div class="empty">Choose a run to render its graph.</div>
          </div>
        </section>

        <section class="detail-stack">
          <section class="panel detail-panel">
            <div class="panel-head">
              <div>
                <h3 class="panel-title">Selected Node</h3>
                <div class="panel-subtle">Click a node in the graph to inspect its neighbors and payload.</div>
              </div>
            </div>
            <div id="node-summary" class="adjacency-shell">
              <div class="empty">No node selected.</div>
            </div>
            <pre id="node-detail" class="codeblock">No node selected.</pre>
          </section>

          <section class="panel detail-panel">
            <div class="panel-head">
              <div>
                <h3 class="panel-title">Run Signals</h3>
                <div class="panel-subtle">Routing, validation, timing, and topology digests for the selected run.</div>
              </div>
            </div>
            <div id="signal-shell" class="empty">No run selected.</div>
          </section>

          <section class="panel detail-panel">
            <div class="panel-head">
              <div>
                <h3 class="panel-title">Agent Metrics</h3>
                <div class="panel-subtle">Per-agent step volume and timing from persisted observability.</div>
              </div>
            </div>
            <div id="agent-metrics-shell" class="empty">No run selected.</div>
          </section>

          <section class="panel detail-panel">
            <div class="panel-head">
              <div>
                <h3 class="panel-title">Failures</h3>
                <div class="panel-subtle">Recent workflow or step-level failures recorded during the run.</div>
              </div>
            </div>
            <div id="failure-shell" class="empty">No run selected.</div>
          </section>

          <section class="panel detail-panel">
            <div class="panel-head">
              <div>
                <h3 class="panel-title">Timeline</h3>
                <div class="panel-subtle">Checkpoint history from the run store.</div>
              </div>
            </div>
            <div id="timeline-shell" class="empty">No run selected.</div>
          </section>

          <section class="panel detail-panel">
            <div class="panel-head">
              <div>
                <h3 class="panel-title">Mermaid</h3>
                <div class="panel-subtle">Raw diagram text for export or debugging.</div>
              </div>
            </div>
            <pre id="mermaid-detail" class="codeblock">No run selected.</pre>
          </section>
        </section>
      </section>
    </main>
  </div>

  <script>
    const initialParams = new URLSearchParams(window.location.search);
    const state = {{
      runs: [],
      filteredRuns: [],
      selectedRunId: initialParams.get('run_id') || null,
      selectedNodeId: initialParams.get('node_id') || null,
      selectedVisualization: null,
      autoRefreshHandle: null,
      autoRefreshIntervalMs: 5000,
    }};

    const kindStyles = {{
      workflow: {{ fill: getCss('--workflow'), stroke: '#2563eb' }},
      attempt: {{ fill: getCss('--attempt'), stroke: '#d97706' }},
      step: {{ fill: getCss('--step'), stroke: '#16a34a' }},
      group_step: {{ fill: getCss('--step'), stroke: '#16a34a' }},
      validator: {{ fill: getCss('--validator'), stroke: '#dc2626' }},
      reducer: {{ fill: getCss('--reducer'), stroke: '#0891b2' }},
      router: {{ fill: getCss('--router'), stroke: '#9333ea' }},
      replan: {{ fill: getCss('--replan'), stroke: '#ca8a04' }},
      clarification: {{ fill: getCss('--clarification'), stroke: '#dc2626' }},
      default: {{ fill: getCss('--default'), stroke: '#78716c' }},
    }};

    function getCss(name) {{
      return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }}

    function formatCount(value, suffix) {{
      const safe = value == null ? 0 : value;
      return `${{safe}} ${{suffix}}`;
    }}

    function formatDuration(value) {{
      if (value == null || value === '' || Number.isNaN(Number(value))) {{
        return 'n/a';
      }}
      const ms = Number(value);
      if (ms < 1000) return `${{ms.toFixed(2)}} ms`;
      return `${{(ms / 1000).toFixed(2)}} s`;
    }}

    function escapeHtml(value) {{
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }}

    function setUrlState() {{
      const url = new URL(window.location.href);
      if (state.selectedRunId) {{
        url.searchParams.set('run_id', state.selectedRunId);
      }} else {{
        url.searchParams.delete('run_id');
      }}
      if (state.selectedNodeId) {{
        url.searchParams.set('node_id', state.selectedNodeId);
      }} else {{
        url.searchParams.delete('node_id');
      }}
      window.history.replaceState({{}}, '', url.toString());
    }}

    function currentGraphView() {{
      return state.selectedVisualization?.graph_view || {{ nodes: [], edges: [], width: 960, height: 320 }};
    }}

    function currentGraphIndex() {{
      return state.selectedVisualization?.graph_index || {{
        nodes: {{}},
        incoming: {{}},
        outgoing: {{}},
        counts: {{ kinds: {{}}, statuses: {{}}, agents: {{}} }},
      }};
    }}

    function currentSummary() {{
      return state.selectedVisualization?.summary || {{}};
    }}

    async function fetchJson(path) {{
      const response = await fetch(path);
      if (!response.ok) {{
        throw new Error(`Request failed: ${{response.status}}`);
      }}
      return response.json();
    }}

    function clearAutoRefresh() {{
      if (state.autoRefreshHandle) {{
        clearTimeout(state.autoRefreshHandle);
        state.autoRefreshHandle = null;
      }}
    }}

    function scheduleAutoRefresh() {{
      clearAutoRefresh();
      const summary = currentSummary();
      const autoRefreshRequested = document.getElementById('auto-refresh')?.checked;
      if (!state.selectedRunId || (!autoRefreshRequested && summary.status !== 'running')) {{
        return;
      }}
      state.autoRefreshHandle = window.setTimeout(async () => {{
        try {{
          await loadRuns({{ skipSelection: true }});
          await refreshSelectedRun({{ preserveNode: true, silent: true }});
        }} catch (error) {{
          console.error('auto-refresh failed', error);
        }} finally {{
          scheduleAutoRefresh();
        }}
      }}, state.autoRefreshIntervalMs);
    }}

    async function loadRuns(options = {{}}) {{
      const status = document.getElementById('run-status').value;
      const agent = document.getElementById('run-agent').value.trim();
      const hasErrors = document.getElementById('run-errors').value;
      const query = new URLSearchParams();
      query.set('limit', '150');
      if (status) query.set('status', status);
      if (agent) query.set('agent', agent);
      if (hasErrors) query.set('has_errors', hasErrors);
      const payload = await fetchJson(`/api/runs?${{query.toString()}}`);
      state.runs = Array.isArray(payload.runs) ? payload.runs : [];
      applyRunFilter();
      if (options.skipSelection) {{
        renderRunList();
        return;
      }}
      const preferredRunId = options.preferredRunId || state.selectedRunId || initialParams.get('run_id');
      const preferredExists = preferredRunId && state.runs.some((run) => run.run_id === preferredRunId);
      if (preferredExists) {{
        await selectRun(preferredRunId, {{ preserveNode: true }});
        return;
      }}
      if (!state.selectedRunId && state.filteredRuns.length) {{
        await selectRun(state.filteredRuns[0].run_id);
      }} else {{
        renderRunList();
      }}
    }}

    function applyRunFilter() {{
      const needle = document.getElementById('run-search').value.trim().toLowerCase();
      state.filteredRuns = state.runs.filter((run) => {{
        if (!needle) return true;
        const haystack = [
          run.run_id,
          run.task,
          run.status,
          run.selected_option_id,
          run.active_step_id,
        ].join(' ').toLowerCase();
        return haystack.includes(needle);
      }});
      renderRunList();
    }}

    function renderRunList() {{
      const container = document.getElementById('run-list');
      if (!state.filteredRuns.length) {{
        container.innerHTML = '<div class="empty">No persisted runs matched the current filters.</div>';
        return;
      }}
      container.innerHTML = state.filteredRuns.map((run) => {{
        const active = run.run_id === state.selectedRunId ? 'active' : '';
        const activeStep = run.active_step_id ? `<span class="chip">active ${{
          escapeHtml(run.active_step_id)
        }}</span>` : '';
        const errorChip = Number(run.error_count || 0) > 0
          ? `<span class="chip">${{escapeHtml(String(run.error_count))}} errors</span>`
          : '';
        const durationChip = run.wall_clock_duration_ms != null
          ? `<span class="chip">${{escapeHtml(formatDuration(run.wall_clock_duration_ms))}}</span>`
          : '';
        return `
          <button class="run-card ${{active}}" data-run-id="${{escapeHtml(run.run_id)}}" type="button">
            <div class="run-id">${{escapeHtml(run.run_id)}}</div>
            <div class="run-task">${{escapeHtml(run.task || '(no task)')}}</div>
            <div class="run-meta">
              <span class="chip">${{escapeHtml(run.status || 'unknown')}}</span>
              <span class="chip">${{formatCount(run.attempt_count || 0, 'attempts')}}</span>
              ${{durationChip}}
              ${{errorChip}}
              ${{activeStep}}
            </div>
          </button>
        `;
      }}).join('');
      container.querySelectorAll('.run-card').forEach((button) => {{
        button.addEventListener('click', () => selectRun(button.dataset.runId));
      }});
    }}

    async function refreshSelectedRun(options = {{}}) {{
      if (!state.selectedRunId) {{
        return;
      }}
      if (!options.silent) {{
        document.getElementById('hero-title').textContent = 'Loading run…';
      }}
      const visualization = await fetchJson(`/api/runs/${{encodeURIComponent(state.selectedRunId)}}`);
      state.selectedVisualization = visualization;
      const graphIndex = currentGraphIndex();
      if (!options.preserveNode || !state.selectedNodeId || !graphIndex.nodes[state.selectedNodeId]) {{
        state.selectedNodeId = options.preserveNode && state.selectedNodeId && graphIndex.nodes[state.selectedNodeId]
          ? state.selectedNodeId
          : null;
      }}
      syncGraphFilters(graphIndex);
      renderVisualization();
    }}

    async function selectRun(runId, options = {{}}) {{
      state.selectedRunId = runId;
      if (!options.preserveNode) {{
        state.selectedNodeId = null;
      }}
      setUrlState();
      renderRunList();
      await refreshSelectedRun({{ preserveNode: options.preserveNode, silent: false }});
    }}

    function renderVisualization() {{
      const payload = state.selectedVisualization;
      if (!payload) return;
      const summary = payload.summary || {{}};
      const observability = payload.observability || {{}};
      document.getElementById('hero-title').textContent = summary.task || payload.run_id || 'Persisted run';
      document.getElementById('hero-subtitle').textContent =
        `Run ${payload.run_id} | last stage: ${summary.last_stage || 'unknown'} | selected option: ${summary.selected_option_id || 'n/a'}`;
      document.getElementById('hero-status').textContent = summary.status || 'unknown';
      renderSummary(summary, observability);
      renderLegend(currentGraphIndex());
      renderSignals(summary, observability, payload.graph_index || {{}});
      renderAgentMetrics(observability.agent_metrics || []);
      renderFailures(observability.failures || []);
      renderGraph(currentGraphView());
      renderTimeline(observability.timeline?.stages || payload.timeline || []);
      document.getElementById('mermaid-detail').textContent = payload.graph_mermaid || 'No Mermaid diagram stored.';
      setUrlState();
      scheduleAutoRefresh();
    }}

    function syncGraphFilters(graphIndex) {{
      const kindSelect = document.getElementById('node-kind-filter');
      const statusSelect = document.getElementById('node-status-filter');
      const currentKind = kindSelect.value;
      const currentStatus = statusSelect.value;
      const kindOptions = Object.keys(graphIndex?.counts?.kinds || {{}}).sort();
      const statusOptions = Object.keys(graphIndex?.counts?.statuses || {{}}).sort();
      kindSelect.innerHTML = ['<option value="">All node kinds</option>']
        .concat(kindOptions.map((kind) => `<option value="${{escapeHtml(kind)}}">${{escapeHtml(kind)}}</option>`))
        .join('');
      statusSelect.innerHTML = ['<option value="">All node statuses</option>']
        .concat(statusOptions.map((status) => `<option value="${{escapeHtml(status)}}">${{escapeHtml(status)}}</option>`))
        .join('');
      if (kindOptions.includes(currentKind)) {{
        kindSelect.value = currentKind;
      }}
      if (statusOptions.includes(currentStatus)) {{
        statusSelect.value = currentStatus;
      }}
    }}

    function renderLegend(graphIndex) {{
      const shell = document.getElementById('graph-legend');
      const kinds = Object.entries(graphIndex?.counts?.kinds || {{}});
      if (!kinds.length) {{
        shell.innerHTML = '<div class="empty">No node kinds available for this run.</div>';
        return;
      }}
      shell.innerHTML = kinds.map(([kind, count]) => {{
        const style = kindStyles[kind] || kindStyles.default;
        return `
          <div class="legend-chip">
            <span class="legend-swatch" style="background:${{style.fill}}; border-color:${{style.stroke}}"></span>
            <span>${{escapeHtml(kind)}}</span>
            <strong>${{escapeHtml(String(count))}}</strong>
          </div>
        `;
      }}).join('');
    }}

    function renderSummary(summary, observability) {{
      const agents = Array.isArray(summary.agents) ? summary.agents : [];
      const slowestSteps = Array.isArray(observability.slowest_steps) ? observability.slowest_steps : [];
      const validationCounts = observability.validation_counts || {{}};
      const stepValidationCount =
        Number(summary.step_validation_count || 0) + Number(summary.workflow_validation_count || 0);
      const cards = [
        {{
          label: 'Status',
          value: summary.status || 'unknown',
          detail: `Run ${summary.run_id || ''}`,
        }},
        {{
          label: 'Attempts',
          value: String(summary.attempt_count || 0),
          detail: `${summary.completed_attempt_count || 0} completed`,
        }},
        {{
          label: 'Graph',
          value: `${summary.graph_node_count || 0} nodes`,
          detail: `${summary.graph_edge_count || 0} edges`,
        }},
        {{
          label: 'Timing',
          value: formatDuration(summary.wall_clock_duration_ms),
          detail: slowestSteps.length
            ? `Slowest step: ${slowestSteps[0].step_id} at ${formatDuration(slowestSteps[0].duration_ms)}`
            : `Step total: ${formatDuration(summary.step_total_duration_ms)}`,
        }},
        {{
          label: 'Errors',
          value: String(summary.error_count || 0),
          detail: summary.has_errors ? 'Run recorded failures or invalid attempts.' : 'No recorded failures.',
        }},
        {{
          label: 'Validation',
          value: String(stepValidationCount || 0),
          detail: `Workflow: ${validationCounts.workflow ? JSON.stringify(validationCounts.workflow) : '{}'} | Step: ${validationCounts.step ? JSON.stringify(validationCounts.step) : '{}'}`,
        }},
        {{
          label: 'Agents',
          value: String(summary.agent_count || agents.length || 0),
          detail: agents.length ? agents.join(', ') : 'No agent metrics stored.',
        }},
        {{
          label: 'Recovery',
          value: summary.resumable ? 'Resumable' : (summary.replayable ? 'Replayable' : 'Final'),
          detail: summary.active_step_id ? `Active step: ${summary.active_step_id}` : (summary.terminal_event || 'No terminal event'),
        }},
      ];
      document.getElementById('summary-grid').innerHTML = cards.map((card) => `
        <article class="summary-card">
          <div class="label">${{escapeHtml(card.label)}}</div>
          <div class="value">${{escapeHtml(card.value)}}</div>
          <div class="detail">${{escapeHtml(card.detail)}}</div>
        </article>
      `).join('');
    }}

    function neighborhoodNodeIds(nodeId, graphIndex) {{
      const ids = new Set();
      if (!nodeId) return ids;
      ids.add(nodeId);
      for (const edge of graphIndex?.incoming?.[nodeId] || []) {{
        ids.add(edge.node_id);
      }}
      for (const edge of graphIndex?.outgoing?.[nodeId] || []) {{
        ids.add(edge.node_id);
      }}
      return ids;
    }}

    function filteredGraphView(graphView) {{
      const graphIndex = currentGraphIndex();
      const search = document.getElementById('node-search').value.trim().toLowerCase();
      const kind = document.getElementById('node-kind-filter').value;
      const status = document.getElementById('node-status-filter').value;
      const focusSelection = document.getElementById('focus-neighborhood').checked;
      const nodes = Array.isArray(graphView.nodes) ? graphView.nodes : [];
      const edges = Array.isArray(graphView.edges) ? graphView.edges : [];
      let visibleIds = new Set(
        nodes
          .filter((node) => {{
            const summary = graphIndex.nodes?.[node.node_id] || {{}};
            if (kind && (summary.kind || node.kind) !== kind) {{
              return false;
            }}
            if (status && (summary.status || node.status) !== status) {{
              return false;
            }}
            if (!search) {{
              return true;
            }}
            const haystack = [
              node.node_id,
              node.title,
              node.subtitle,
              summary.agent_name,
              summary.task,
              summary.kind,
            ]
              .join(' ')
              .toLowerCase();
            return haystack.includes(search);
          }})
          .map((node) => node.node_id)
      );

      if (!search && !kind && !status) {{
        visibleIds = new Set(nodes.map((node) => node.node_id));
      }}

      if (focusSelection && state.selectedNodeId) {{
        const relatedIds = neighborhoodNodeIds(state.selectedNodeId, graphIndex);
        const focusedIds = new Set();
        for (const nodeId of relatedIds) {{
          if (visibleIds.has(nodeId) || (!search && !kind && !status)) {{
            focusedIds.add(nodeId);
          }}
        }}
        if (!focusedIds.size) {{
          for (const nodeId of relatedIds) {{
            focusedIds.add(nodeId);
          }}
        }}
        visibleIds = focusedIds;
      }}

      const filteredNodes = nodes.filter((node) => visibleIds.has(node.node_id));
      const filteredEdges = edges.filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target));
      return {{ nodes: filteredNodes, edges: filteredEdges }};
    }}

    function renderGraph(graphView) {{
      const shell = document.getElementById('graph-shell');
      const filtered = filteredGraphView(graphView);
      const nodes = filtered.nodes;
      const edges = filtered.edges;
      const totalNodes = Array.isArray(graphView.nodes) ? graphView.nodes.length : 0;
      const totalEdges = Array.isArray(graphView.edges) ? graphView.edges.length : 0;
      document.getElementById('graph-stats').textContent =
        `${nodes.length}/${{totalNodes}} nodes | ${edges.length}/${{totalEdges}} edges`;
      if (!nodes.length) {{
        shell.innerHTML = '<div class="empty">No graph nodes matched the current filters.</div>';
        document.getElementById('node-summary').innerHTML = '<div class="empty">No node selected.</div>';
        document.getElementById('node-detail').textContent = 'No node selected.';
        return;
      }}

      const graphIndex = currentGraphIndex();
      const relatedIds = neighborhoodNodeIds(state.selectedNodeId, graphIndex);
      const svgParts = [];
      svgParts.push(`<svg viewBox="0 0 ${graphView.width || 1200} ${graphView.height || 400}" role="img" aria-label="Workflow graph">`);
      svgParts.push('<defs><marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><polygon points="0 0, 10 4, 0 8" fill="rgba(85,70,51,0.4)"></polygon></marker></defs>');

      for (const edge of edges) {{
        svgParts.push(`<path class="edge" d="${edge.path}" marker-end="url(#arrowhead)"></path>`);
        if (edge.relation) {{
          svgParts.push(`<text class="edge-label" x="${edge.label_x}" y="${edge.label_y}">${escapeHtml(edge.relation)}</text>`);
        }}
      }}

      for (const node of nodes) {{
        const style = kindStyles[node.kind] || kindStyles.default;
        const active = node.node_id === state.selectedNodeId ? 'active' : '';
        const related = node.node_id !== state.selectedNodeId && relatedIds.has(node.node_id) ? 'related' : '';
        const title = escapeHtml(node.title || node.node_id);
        const subtitle = escapeHtml(node.subtitle || '');
        svgParts.push(`
          <g class="node-card ${active} ${related}" data-node-id="${escapeHtml(node.node_id)}" tabindex="0" role="button" aria-label="${title}">
            <rect x="${node.x}" y="${node.y}" rx="18" ry="18" width="${node.width}" height="${node.height}" fill="${style.fill}" stroke="${style.stroke}"></rect>
            <text class="node-title" x="${node.x + 16}" y="${node.y + 28}">${title}</text>
            <text class="node-subtitle" x="${node.x + 16}" y="${node.y + 50}">${subtitle}</text>
          </g>
        `);
      }}

      svgParts.push('</svg>');
      shell.innerHTML = svgParts.join('');
      shell.querySelectorAll('.node-card').forEach((nodeElement) => {{
        const activate = () => {{
          state.selectedNodeId = nodeElement.dataset.nodeId;
          setUrlState();
          renderNodeDetail();
          renderGraph(graphView);
        }};
        nodeElement.addEventListener('click', activate);
        nodeElement.addEventListener('keydown', (event) => {{
          if (event.key === 'Enter' || event.key === ' ') {{
            event.preventDefault();
            activate();
          }}
        }});
      }});
      renderNodeDetail();
    }}

    function renderNodeDetail() {{
      const payload = state.selectedVisualization;
      const detail = document.getElementById('node-detail');
      const summaryShell = document.getElementById('node-summary');
      if (!payload || !payload.graph_view || !Array.isArray(payload.graph_view.nodes)) {{
        summaryShell.innerHTML = '<div class="empty">No node selected.</div>';
        detail.textContent = 'No node selected.';
        return;
      }}
      const match = payload.graph_view.nodes.find((node) => node.node_id === state.selectedNodeId);
      if (!match) {{
        summaryShell.innerHTML = '<div class="empty">No node selected.</div>';
        detail.textContent = 'No node selected.';
        return;
      }}
      const graphIndex = currentGraphIndex();
      const nodeSummary = graphIndex.nodes?.[match.node_id] || {{}};
      const incoming = graphIndex.incoming?.[match.node_id] || [];
      const outgoing = graphIndex.outgoing?.[match.node_id] || [];
      const summaryCards = [
        {{
          label: 'Kind',
          value: nodeSummary.kind || match.kind || 'node',
          detail: nodeSummary.status || match.status || 'unknown',
        }},
        {{
          label: 'Depth',
          value: String(nodeSummary.depth ?? match.depth ?? 0),
          detail: nodeSummary.step_id || nodeSummary.agent_name || 'graph node',
        }},
        {{
          label: 'Attempt',
          value: String(nodeSummary.attempt ?? 'n/a'),
          detail: nodeSummary.agent_name || 'no agent',
        }},
        {{
          label: 'Edges',
          value: `${incoming.length} in / ${outgoing.length} out`,
          detail: nodeSummary.task || match.title || 'node',
        }},
      ];
      summaryShell.innerHTML = `
        <div class="signal-grid">
          ${summaryCards.map((card) => `
            <article class="signal-card">
              <div class="label">${escapeHtml(card.label)}</div>
              <div class="value">${escapeHtml(card.value)}</div>
              <div class="detail">${escapeHtml(card.detail)}</div>
            </article>
          `).join('')}
        </div>
        <div class="adjacency-group">
          <div class="adjacency-title">Incoming</div>
          <div class="adjacency-list">
            ${incoming.length
              ? incoming.map((edge) => `<span class="adjacency-pill"><strong>${escapeHtml(edge.node_id)}</strong><span>${escapeHtml(edge.relation || 'edge')}</span></span>`).join('')
              : '<span class="adjacency-pill">No incoming edges</span>'}
          </div>
        </div>
        <div class="adjacency-group">
          <div class="adjacency-title">Outgoing</div>
          <div class="adjacency-list">
            ${outgoing.length
              ? outgoing.map((edge) => `<span class="adjacency-pill"><strong>${escapeHtml(edge.node_id)}</strong><span>${escapeHtml(edge.relation || 'edge')}</span></span>`).join('')
              : '<span class="adjacency-pill">No outgoing edges</span>'}
          </div>
        </div>
      `;
      detail.textContent = JSON.stringify(match.data, null, 2);
    }}

    function renderSignals(summary, observability, graphIndex) {{
      const shell = document.getElementById('signal-shell');
      const routingCounts = observability.routing_action_counts || {{}};
      const validationCounts = observability.validation_counts || {{}};
      const stageCounts = observability.timeline?.stage_counts || {{}};
      const validationDigest = [
        ...Object.entries(validationCounts.workflow || {{}}).map(([verdict, count]) => `workflow ${verdict}: ${count}`),
        ...Object.entries(validationCounts.step || {{}}).map(([verdict, count]) => `step ${verdict}: ${count}`),
      ].join(' | ') || 'No validation metrics recorded.';
      const routingDigest = Object.entries(routingCounts)
        .sort((left, right) => Number(right[1]) - Number(left[1]))
        .slice(0, 3)
        .map(([action, count]) => `${action}: ${count}`)
        .join(' | ') || 'No routing actions recorded.';
      const stageDigest = Object.entries(stageCounts)
        .sort((left, right) => Number(right[1]) - Number(left[1]))
        .slice(0, 3)
        .map(([stage, count]) => `${stage}: ${count}`)
        .join(' | ') || 'No timeline stages recorded.';
      const cards = [
        {{
          label: 'Topology',
          value: `${summary.graph_node_count || 0} nodes`,
          detail: `${summary.graph_edge_count || 0} edges | root ${graphIndex.root_node_id || 'n/a'}`,
        }},
        {{
          label: 'Routing',
          value: String(Object.keys(routingCounts).length),
          detail: routingDigest,
        }},
        {{
          label: 'Validation',
          value: String((summary.step_validation_count || 0) + (summary.workflow_validation_count || 0)),
          detail: validationDigest,
        }},
        {{
          label: 'Timeline',
          value: String(summary.timeline_entries || 0),
          detail: stageDigest,
        }},
      ];
      shell.innerHTML = `
        <div class="signal-grid">
          ${cards.map((card) => `
            <article class="signal-card">
              <div class="label">${escapeHtml(card.label)}</div>
              <div class="value">${escapeHtml(card.value)}</div>
              <div class="detail">${escapeHtml(card.detail)}</div>
            </article>
          `).join('')}
        </div>
      `;
    }}

    function renderAgentMetrics(agentMetrics) {{
      const shell = document.getElementById('agent-metrics-shell');
      if (!Array.isArray(agentMetrics) || !agentMetrics.length) {{
        shell.innerHTML = '<div class="empty">No agent metrics stored for this run.</div>';
        return;
      }}
      shell.innerHTML = `
        <table class="metrics-table">
          <thead>
            <tr>
              <th>Agent</th>
              <th>Steps</th>
              <th>Total</th>
              <th>Max</th>
            </tr>
          </thead>
          <tbody>
            ${agentMetrics.slice(0, 10).map((metric) => `
              <tr>
                <td>${escapeHtml(metric.agent || 'unknown')}</td>
                <td>${escapeHtml(String(metric.step_count || 0))}</td>
                <td>${escapeHtml(formatDuration(metric.total_duration_ms))}</td>
                <td>${escapeHtml(formatDuration(metric.max_duration_ms))}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;
    }}

    function renderFailures(failures) {{
      const shell = document.getElementById('failure-shell');
      if (!Array.isArray(failures) || !failures.length) {{
        shell.innerHTML = '<div class="empty">No failures were recorded for this run.</div>';
        return;
      }}
      shell.innerHTML = `
        <div class="failure-list">
          ${failures.slice(0, 8).map((failure) => `
            <article class="failure-card">
              <div class="title">${escapeHtml(failure.scope || 'failure')} ${escapeHtml(failure.step_id || failure.option_id || '')}</div>
              <div class="meta">${escapeHtml(failure.status || 'unknown')} | attempt ${escapeHtml(String(failure.attempt ?? 'n/a'))}</div>
              <div class="reason">${escapeHtml(failure.reason || failure.error || 'No failure reason recorded.')}</div>
            </article>
          `).join('')}
        </div>
      `;
    }}

    function renderTimeline(timeline) {{
      const shell = document.getElementById('timeline-shell');
      if (!Array.isArray(timeline) || !timeline.length) {{
        shell.innerHTML = '<div class="empty">No timeline checkpoints stored for this run.</div>';
        return;
      }}
      shell.innerHTML = `
        <table class="timeline">
          <thead>
            <tr>
              <th>Stage</th>
              <th>Status</th>
              <th>Delta</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody>
            ${timeline.map((entry) => `
              <tr>
                <td>${escapeHtml(entry.stage || '')}</td>
                <td>${escapeHtml(entry.status || '')}</td>
                <td>${escapeHtml(formatDuration(entry.delta_ms_from_previous))}</td>
                <td>${escapeHtml(entry.updated_at || '')}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;
    }}

    document.getElementById('run-search').addEventListener('input', applyRunFilter);
    document.getElementById('run-status').addEventListener('change', loadRuns);
    document.getElementById('run-agent').addEventListener('input', loadRuns);
    document.getElementById('run-errors').addEventListener('change', loadRuns);
    document.getElementById('node-search').addEventListener('input', () => renderGraph(currentGraphView()));
    document.getElementById('node-kind-filter').addEventListener('change', () => renderGraph(currentGraphView()));
    document.getElementById('node-status-filter').addEventListener('change', () => renderGraph(currentGraphView()));
    document.getElementById('focus-neighborhood').addEventListener('change', () => renderGraph(currentGraphView()));
    document.getElementById('auto-refresh').addEventListener('change', scheduleAutoRefresh);
    loadRuns({{ preferredRunId: state.selectedRunId || undefined }}).catch((error) => {{
      document.getElementById('run-list').innerHTML = `<div class="empty">Failed to load runs: ${escapeHtml(error.message)}</div>`;
    }});
  </script>
</body>
</html>"""
    return template.replace("__BASE_DIR__", safe_base_dir).replace("{{", "{").replace("}}", "}")


def create_run_visualizer_app(run_store: RunStore | None = None) -> FastAPI:
    store = run_store or RunStore()
    app = FastAPI(title="OpenFabric Run Visualizer")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(render_run_visualizer_html(base_dir=str(store.base_dir)))

    @app.get("/api/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok", "base_dir": str(store.base_dir)})

    @app.get("/api/runs")
    def api_runs(
        limit: int = 50,
        status: str = "",
        task_contains: str = "",
        agent: str = "",
        has_errors: str = "",
        min_duration_ms: float | None = None,
        max_duration_ms: float | None = None,
        slow_step_ms: float | None = None,
    ) -> JSONResponse:
        error_filter = None
        if str(has_errors).strip().lower() in {"1", "true", "yes", "errors"}:
            error_filter = True
        payload = list_run_visualizations(
            store,
            limit=max(1, min(limit, 500)),
            status=status or None,
            task_contains=task_contains or None,
            agent=agent or None,
            has_errors=error_filter,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            slow_step_ms=slow_step_ms,
        )
        return JSONResponse(payload)

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: str) -> JSONResponse:
        try:
            payload = load_run_visualization(store, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc
        return JSONResponse(payload)

    @app.get("/api/runs/{run_id}/graph", response_model=None)
    def api_run_graph(run_id: str, format: str = "view") -> PlainTextResponse | JSONResponse:
        try:
            payload = load_run_graph_payload(store, run_id, format=format)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if isinstance(payload, str):
            return PlainTextResponse(payload)
        return JSONResponse(payload)

    @app.get("/api/runs/{run_id}/observability")
    def api_run_observability(run_id: str) -> JSONResponse:
        try:
            payload = load_run_observability_payload(store, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc
        return JSONResponse(payload)

    return app


def serve_run_visualizer(
    *,
    host: str = DEFAULT_UI_HOST,
    port: int = DEFAULT_UI_PORT,
    run_store: RunStore | None = None,
) -> None:
    import uvicorn

    app = create_run_visualizer_app(run_store=run_store)
    uvicorn.run(app, host=host, port=port)
