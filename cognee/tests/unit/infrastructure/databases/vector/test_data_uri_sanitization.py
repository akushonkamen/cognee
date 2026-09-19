"""Embedding input sanitizer neutralizes inline data URIs.

NVIDIA NIM's embedding endpoint scans input text for "data:image/...;base64"
payloads and rejects the whole request with 400 "image inputs require VLM
serving" — a plain string quoting such a URI (e.g. a memory summary of a
conversation about a chart-generating app) must be embeddable.
"""

from cognee.infrastructure.databases.vector.embeddings.utils import (
    is_embeddable,
    sanitize_embedding_text_inputs,
)


def test_data_uri_in_text_is_neutralized():
    text = 'route returns "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==" with status 200'
    (out,) = sanitize_embedding_text_inputs([text])

    assert "data:image" not in out
    assert "[embedded data uri]" in out
    assert is_embeddable(out)


def test_template_placeholder_data_uri_is_neutralized():
    """NIM flags on the URI prefix alone — even a code template like
    'data:image/png;base64,{img_base64}' must be neutralized."""
    text = 'return jsonify({"image": f"data:image/png;base64,{img_base64}"}), 200'
    (out,) = sanitize_embedding_text_inputs([text])

    assert "data:image" not in out


def test_plain_text_survives_sanitization():
    text = "a plain sentence mentioning base64 image encoding, no URI"
    assert sanitize_embedding_text_inputs([text]) == [text]


def test_non_string_inputs_still_get_dummy():
    assert sanitize_embedding_text_inputs([None, "", "ok"]) == [".", ".", "ok"]
