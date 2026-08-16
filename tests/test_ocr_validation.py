from src.ocr.paddle import RawOCRLine
from src.ocr.validation import parse_answer_line
from src.schemas import BoundingBox


def _line(text: str, confidence: float = 0.95) -> RawOCRLine:
    return RawOCRLine(text=text, confidence=confidence, bbox=BoundingBox(x=0, y=0, width=10, height=10))


def test_parses_dot_separator():
    result = parse_answer_line(_line("9. A"), num_questions=20)
    assert result.question_number == 9
    assert result.detected_answer == "A"
    assert not result.needs_escalation


def test_parses_paren_and_dash_noise():
    assert parse_answer_line(_line("12) C"), 20).detected_answer == "C"
    assert parse_answer_line(_line("3 - D"), 20).detected_answer == "D"
    assert parse_answer_line(_line("5B"), 20).detected_answer == "B"


def test_out_of_range_question_number_not_accepted():
    result = parse_answer_line(_line("99. A"), num_questions=20)
    assert result.question_number is None
    assert result.needs_escalation


def test_invalid_letter_not_accepted():
    result = parse_answer_line(_line("4. E"), num_questions=20)
    assert result.detected_answer is None
    assert result.needs_escalation


def test_low_confidence_escalates_even_if_parseable():
    result = parse_answer_line(_line("1. A", confidence=0.5), num_questions=20)
    assert result.needs_escalation
