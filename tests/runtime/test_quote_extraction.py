from __future__ import annotations

from aor_runtime.runtime.text_extract import extract_quoted_content


def test_extract_exact_text_single_quotes() -> None:
    assert extract_quoted_content("Write the exact text 'quiet library' to /tmp/a.txt and return it") == "quiet library"


def test_extract_content_double_quotes() -> None:
    assert extract_quoted_content('Create /tmp/a.txt with exact content "hello world" and return the content only') == "hello world"


def test_extract_containing_phrase() -> None:
    assert extract_quoted_content("Create /tmp/a.txt containing 'kind words' and return it") == "kind words"


def test_extract_write_phrase() -> None:
    assert extract_quoted_content("Save 'alpha beta' to /tmp/a.txt and return the saved file") == "alpha beta"
