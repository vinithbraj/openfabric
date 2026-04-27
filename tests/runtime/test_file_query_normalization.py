from __future__ import annotations

from aor_runtime.runtime.file_query import normalize_file_query


def test_top_level_synonyms_map_to_non_recursive() -> None:
    assert normalize_file_query("list top-level txt files in /tmp/example", "/tmp/example").recursive is False
    assert normalize_file_query("list txt files directly in /tmp/example", "/tmp/example").recursive is False
    assert normalize_file_query("list txt files not nested in /tmp/example", "/tmp/example").recursive is False
    assert normalize_file_query("count direct txt files in /tmp/example", "/tmp/example").recursive is False


def test_recursive_synonyms_map_to_recursive() -> None:
    assert normalize_file_query("count txt files under /tmp/example", "/tmp/example").recursive is True
    assert normalize_file_query("count txt files recursively in /tmp/example", "/tmp/example").recursive is True
    assert normalize_file_query("count txt files anywhere below /tmp/example", "/tmp/example").recursive is True


def test_txt_phrase_variants_normalize_to_txt_glob() -> None:
    assert normalize_file_query("count txt files in /tmp/example", "/tmp/example").pattern == "*.txt"
    assert normalize_file_query("count .txt files in /tmp/example", "/tmp/example").pattern == "*.txt"
    assert normalize_file_query("count text files in /tmp/example", "/tmp/example").pattern == "*.txt"
    assert normalize_file_query("count files ending in txt in /tmp/example", "/tmp/example").pattern == "*.txt"


def test_filenames_only_sets_name_path_style() -> None:
    assert normalize_file_query("return matching filenames only from /tmp/example", "/tmp/example").path_style == "name"
    assert normalize_file_query("return txt filenames from /tmp/example", "/tmp/example").path_style == "name"


def test_content_search_prompt_does_not_infer_bogus_extension_pattern() -> None:
    query = normalize_file_query(
        "Find the files in /tmp/example whose contents include weekend and return just the filenames as csv.",
        "/tmp/example",
    )

    assert query.pattern == "*"
