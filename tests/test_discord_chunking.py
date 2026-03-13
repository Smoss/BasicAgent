from utils.text import chunk_message


def test_short_message_no_split():
    msg = "Hello, world!"
    result = chunk_message(msg, limit=1900)
    assert result == [msg]


def test_long_message_splits_on_newlines():
    lines = [f"Line {i}" for i in range(300)]
    msg = "\n".join(lines)
    chunks = chunk_message(msg, limit=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100


def test_single_long_line_splits_by_character():
    line = "x" * 250
    chunks = chunk_message(line, limit=100)
    assert len(chunks) >= 2
    assert chunks[0] == "x" * 100


def test_exact_limit_no_split():
    msg = "a" * 1900
    result = chunk_message(msg, limit=1900)
    assert result == [msg]


def test_custom_limit():
    msg = "a" * 50 + "\n" + "b" * 50
    chunks = chunk_message(msg, limit=60)
    assert len(chunks) == 2
