"""Tests for app.processors.transcribe — mock ASR processor."""

from app.processors.transcribe import mock_transcribe


async def test_mock_transcribe_returns_expected_keys() -> None:
    """Result must contain all documented output fields."""
    result = await mock_transcribe("http://example.com/audio.mp3", language="en")

    assert set(result.keys()) == {
        "transcript",
        "language",
        "duration",
        "confidence",
        "segments",
    }


async def test_mock_transcribe_explicit_language_preserved() -> None:
    """When language is not 'auto', the detected_language equals the input."""
    result = await mock_transcribe("http://example.com/audio.mp3", language="es")
    assert result["language"] == "es"


async def test_mock_transcribe_auto_language_is_valid() -> None:
    """'auto' language resolves to one of the known language codes."""
    known_languages = {"en", "es", "fr"}
    result = await mock_transcribe("http://example.com/audio.mp3", language="auto")
    assert result["language"] in known_languages


async def test_mock_transcribe_confidence_in_range() -> None:
    """Confidence is a float in [0, 1]."""
    result = await mock_transcribe("http://example.com/audio.mp3", language="en")
    confidence = result["confidence"]
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


async def test_mock_transcribe_duration_positive() -> None:
    """Duration is a positive float representing seconds."""
    result = await mock_transcribe("http://example.com/audio.mp3", language="en")
    assert isinstance(result["duration"], float)
    assert result["duration"] > 0.0


async def test_mock_transcribe_segments_list() -> None:
    """Segments is a non-empty list with expected keys."""
    result = await mock_transcribe("http://example.com/audio.mp3", language="en")
    segs = result["segments"]
    assert isinstance(segs, list)
    assert len(segs) >= 1
    for seg in segs:
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg


async def test_mock_transcribe_transcript_contains_url() -> None:
    """The mock transcript includes the audio URL for traceability."""
    url = "http://example.com/my_audio.mp3"
    result = await mock_transcribe(url, language="en")
    assert url in result["transcript"]


async def test_mock_transcribe_fast_with_patched_sleep() -> None:
    """With asyncio.sleep mocked the function completes instantly."""
    import unittest.mock as mock

    with mock.patch("app.processors.transcribe.asyncio.sleep", return_value=None):
        result = await mock_transcribe("http://example.com/a.mp3", language="en")

    assert "transcript" in result
