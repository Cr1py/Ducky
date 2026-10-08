import pytest

from ducky import endphrase

PHRASES = ["have any thoughts ducky"]


def test_strips_tail_phrase():
    text, hit = endphrase.strip_end_phrase(
        "my loop never terminates. Have any thoughts, Ducky?", PHRASES
    )
    assert hit and text == "my loop never terminates"


def test_mid_sentence_does_not_trigger():
    text, hit = endphrase.strip_end_phrase(
        "have any thoughts ducky is what I will say later but not yet", PHRASES
    )
    assert not hit and "later" in text


def test_min_words(conn):
    with pytest.raises(endphrase.PhraseError):
        endphrase.add_phrase(conn, "ducky")
    assert endphrase.add_phrase(conn, "Over to you, duck!") == "over to you duck"
    with pytest.raises(endphrase.PhraseError, match="already"):
        endphrase.add_phrase(conn, "over to you duck")
