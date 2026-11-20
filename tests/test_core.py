from neuroforge.core import make_messages, parse_sentiment


def test_strict_parser():
    assert parse_sentiment(" positive \n") == "positive"
    assert parse_sentiment("negative") == "negative"
    assert parse_sentiment("probably positive") is None


def test_prompt():
    messages = make_messages("Great film.")
    assert len(messages) == 2
    assert "Great film." in messages[1]["content"]
