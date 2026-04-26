from __future__ import annotations

from aor_runtime.runtime.task_graph import ArtifactKind, ArtifactRef, TaskGraph, TaskNode


def test_artifact_kind_enum_values_exist() -> None:
    assert ArtifactKind.TEXT.value == "text"
    assert ArtifactKind.COUNT.value == "count"
    assert ArtifactKind.FILE_MATCHES.value == "file_matches"
    assert ArtifactKind.SHELL_STDOUT.value == "shell_stdout"


def test_artifact_ref_validates() -> None:
    ref = ArtifactRef(name="matches", kind=ArtifactKind.FILE_MATCHES, path="files")
    assert ref.name == "matches"
    assert ref.kind is ArtifactKind.FILE_MATCHES
    assert ref.path == "files"


def test_task_node_validates() -> None:
    node = TaskNode(id="read_1", intent={"kind": "read"}, output=ArtifactRef(name="content", kind=ArtifactKind.TEXT))
    assert node.id == "read_1"
    assert node.output is not None
    assert node.output.kind is ArtifactKind.TEXT


def test_task_graph_validates_multiple_nodes_and_final_output() -> None:
    graph = TaskGraph(
        nodes=[
            TaskNode(id="n1", intent={"kind": "read"}, output=ArtifactRef(name="content", kind=ArtifactKind.TEXT)),
            TaskNode(id="n2", intent={"kind": "transform"}, output=ArtifactRef(name="csv", kind=ArtifactKind.CSV)),
        ],
        final_output=ArtifactRef(name="csv", kind=ArtifactKind.CSV),
    )
    assert len(graph.nodes) == 2
    assert graph.final_output is not None
    assert graph.final_output.kind is ArtifactKind.CSV
