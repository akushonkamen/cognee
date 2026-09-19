import re
from typing import List, Union
from cognee.shared.logging_utils import setup_logging

# Inline data URIs (e.g. a conversation that quotes a
# "data:image/png;base64,..." payload) are rejected by some embedding
# endpoints — NVIDIA NIM scans input text for them and answers 400
# "image inputs require VLM serving" even though the input is a plain
# string. The base64 payload carries no semantic value for the embedding
# anyway, so the sanitizer neutralizes the whole URI.
_DATA_URI_RE = re.compile(
    r"data:[a-z0-9.+-]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE
)

logger = setup_logging()


def is_embeddable(s: str) -> bool:
    """
    Check if input string is embeddable, if not it will be replaced with a dummy value to prevent API errors.
    Empty strings and strings containing only whitespace are not embeddable.
    If input string contains at least one non-whitespace character, it is considered embeddable.
    """
    if not isinstance(s, str):
        return False
    # Strip whitespace to check if the string is empty or only contains spaces
    s = s.strip()
    if len(s) >= 1:
        return True
    logger.debug(
        "Input string was not embeddable. Skipping embedding and using dummy value instead."
    )
    return False


def sanitize_embedding_text_inputs(text: Union[str, List[str]]) -> List[str]:
    """
    Transform invalid/empty inputs into a safe dummy to prevent API 422 embedding errors while
    keeping list length consistent.
    """
    # Ensure we are working with a list
    text_list = [text] if isinstance(text, str) else text
    dummy_value = "."

    cleaned = [_DATA_URI_RE.sub("[embedded data uri]", t) if isinstance(t, str) else t for t in text_list]
    return [t if is_embeddable(t) else dummy_value for t in cleaned]


def handle_embedding_response(
    original_texts: Union[List[str], str], embeddings: List[List[float]], dimensions: int
) -> List[List[float]]:
    """
    Compare the original input strings against the results.
    If the original string was 'junk' that was not embeddable, overwrite its vector with zeros.
    """
    if isinstance(original_texts, str):
        original_texts = [original_texts]

    zero_vector = [0.0] * dimensions
    return [
        embeddings[i] if is_embeddable(original_texts[i]) else zero_vector
        for i in range(len(original_texts))
    ]
