import copy
import html
import json
from collections import defaultdict, deque
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .run_store import RunStore


VISUALIZER_SCHEMA_VERSION = "phase1"
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


def build_run_visualization_payload(inspection: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(inspection, dict):
        raise ValueError("Run inspection payload must be a dict.")
    graph = inspection.get("graph") if isinstance(inspection.get("graph"), dict) else {}
    return {
        "schema_version": VISUALIZER_SCHEMA_VERSION,
        "run_id": str(inspection.get("run_id") or ""),
        "summary": copy.deepcopy(inspection.get("summary")) if isinstance(inspection.get("summary"), dict) else {},
        "state": copy.deepcopy(inspection.get("state")) if isinstance(inspection.get("state"), dict) else {},
        "graph": copy.deepcopy(graph),
        "graph_view": build_graph_view_model(graph),
        "graph_mermaid": str(inspection.get("graph_mermaid") or ""),
        "timeline": copy.deepcopy(inspection.get("timeline")) if isinstance(inspection.get("timeline"), list) else [],
    }


def list_run_visualizations(
    run_store: RunStore,
    *,
    limit: int = 50,
    status: str | None = None,
) -> dict[str, Any]:
    runs = run_store.list_runs(limit=limit, status=status)
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
    }}

    @media (max-width: 720px) {{
      .shell {{
        padding: 14px;
        gap: 14px;
      }}

      .summary-grid {{
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
          <div id="graph-shell" class="graph-shell">
            <div class="empty">Choose a run to render its graph.</div>
          </div>
        </section>

        <section class="detail-stack">
          <section class="panel detail-panel">
            <div class="panel-head">
              <div>
                <h3 class="panel-title">Selected Node</h3>
                <div class="panel-subtle">Click a node in the graph to inspect its payload.</div>
              </div>
            </div>
            <pre id="node-detail" class="codeblock">No node selected.</pre>
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
    const state = {{
      runs: [],
      filteredRuns: [],
      selectedRunId: null,
      selectedNodeId: null,
      selectedVisualization: null,
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

    function escapeHtml(value) {{
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }}

    async function fetchJson(path) {{
      const response = await fetch(path);
      if (!response.ok) {{
        throw new Error(`Request failed: ${{response.status}}`);
      }}
      return response.json();
    }}

    async function loadRuns() {{
      const status = document.getElementById('run-status').value;
      const query = new URLSearchParams();
      query.set('limit', '150');
      if (status) query.set('status', status);
      const payload = await fetchJson(`/api/runs?${{query.toString()}}`);
      state.runs = Array.isArray(payload.runs) ? payload.runs : [];
      applyRunFilter();
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
        return `
          <button class="run-card ${{active}}" data-run-id="${{escapeHtml(run.run_id)}}" type="button">
            <div class="run-id">${{escapeHtml(run.run_id)}}</div>
            <div class="run-task">${{escapeHtml(run.task || '(no task)')}}</div>
            <div class="run-meta">
              <span class="chip">${{escapeHtml(run.status || 'unknown')}}</span>
              <span class="chip">${{formatCount(run.attempt_count || 0, 'attempts')}}</span>
              ${{activeStep}}
            </div>
          </button>
        `;
      }}).join('');
      container.querySelectorAll('.run-card').forEach((button) => {{
        button.addEventListener('click', () => selectRun(button.dataset.runId));
      }});
    }}

    async function selectRun(runId) {{
      state.selectedRunId = runId;
      renderRunList();
      document.getElementById('hero-title').textContent = 'Loading run…';
      const visualization = await fetchJson(`/api/runs/${{encodeURIComponent(runId)}}`);
      state.selectedVisualization = visualization;
      state.selectedNodeId = null;
      renderVisualization();
    }}

    function renderVisualization() {{
      const payload = state.selectedVisualization;
      if (!payload) return;
      const summary = payload.summary || {{}};
      document.getElementById('hero-title').textContent = summary.task || payload.run_id || 'Persisted run';
      document.getElementById('hero-subtitle').textContent =
        `Run ${payload.run_id} | last stage: ${summary.last_stage || 'unknown'} | selected option: ${summary.selected_option_id || 'n/a'}`;
      document.getElementById('hero-status').textContent = summary.status || 'unknown';
      renderSummary(summary);
      renderGraph(payload.graph_view || {{ nodes: [], edges: [] }});
      renderTimeline(payload.timeline || []);
      document.getElementById('mermaid-detail').textContent = payload.graph_mermaid || 'No Mermaid diagram stored.';
    }}

    function renderSummary(summary) {{
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

    function renderGraph(graphView) {{
      const shell = document.getElementById('graph-shell');
      const nodes = Array.isArray(graphView.nodes) ? graphView.nodes : [];
      const edges = Array.isArray(graphView.edges) ? graphView.edges : [];
      document.getElementById('graph-stats').textContent =
        `${nodes.length} nodes | ${edges.length} edges`;
      if (!nodes.length) {{
        shell.innerHTML = '<div class="empty">No graph nodes were stored for this run.</div>';
        document.getElementById('node-detail').textContent = 'No node selected.';
        return;
      }}

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
        const title = escapeHtml(node.title || node.node_id);
        const subtitle = escapeHtml(node.subtitle || '');
        svgParts.push(`
          <g class="node-card ${active}" data-node-id="${escapeHtml(node.node_id)}" tabindex="0" role="button" aria-label="${title}">
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
      if (!payload || !payload.graph_view || !Array.isArray(payload.graph_view.nodes)) {{
        detail.textContent = 'No node selected.';
        return;
      }}
      const match = payload.graph_view.nodes.find((node) => node.node_id === state.selectedNodeId);
      detail.textContent = match ? JSON.stringify(match.data, null, 2) : 'No node selected.';
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
              <th>Attempt</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody>
            ${timeline.map((entry) => `
              <tr>
                <td>${escapeHtml(entry.stage || '')}</td>
                <td>${escapeHtml(entry.status || '')}</td>
                <td>${escapeHtml(String(entry.current_attempt_index ?? ''))}</td>
                <td>${escapeHtml(entry.updated_at || '')}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;
    }}

    document.getElementById('run-search').addEventListener('input', applyRunFilter);
    document.getElementById('run-status').addEventListener('change', loadRuns);
    loadRuns().catch((error) => {{
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
    def api_runs(limit: int = 50, status: str = "") -> JSONResponse:
        payload = list_run_visualizations(store, limit=max(1, min(limit, 500)), status=status or None)
        return JSONResponse(payload)

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: str) -> JSONResponse:
        try:
            payload = load_run_visualization(store, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc
        return JSONResponse(payload)

    @app.get("/api/runs/{run_id}/graph")
    def api_run_graph(run_id: str, format: str = "view") -> JSONResponse | PlainTextResponse:
        try:
            payload = load_run_graph_payload(store, run_id, format=format)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if isinstance(payload, str):
            return PlainTextResponse(payload)
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
