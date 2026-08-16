from src.grading.grader import AnswerKeyEntry, DetectedAnswer, grade_submission
from src.schemas import AnswerState


def _key():
    return [
        AnswerKeyEntry(1, "A", 1.0),
        AnswerKeyEntry(2, "B", 1.0),
        AnswerKeyEntry(3, "C", 2.0),
    ]


def test_all_correct():
    detected = [
        DetectedAnswer(1, "A", AnswerState.CLEAR),
        DetectedAnswer(2, "B", AnswerState.CLEAR),
        DetectedAnswer(3, "C", AnswerState.CLEAR),
    ]
    result = grade_submission(detected, _key())
    assert result.correct_count == 3
    assert result.incorrect_count == 0
    assert result.unanswered_count == 0
    assert result.score == 4.0
    assert result.total_points == 4.0
    assert result.percentage == 100.0


def test_mixed_and_unanswered():
    detected = [
        DetectedAnswer(1, "B", AnswerState.CLEAR),  # incorrect
        DetectedAnswer(2, None, AnswerState.BLANK),  # unanswered
        DetectedAnswer(3, "C", AnswerState.STRUCK_THROUGH),  # struck -> unanswered
    ]
    result = grade_submission(detected, _key())
    assert result.correct_count == 0
    assert result.incorrect_count == 1
    assert result.unanswered_count == 2
    assert result.score == 0.0
    graded_by_q = {g.question_number: g for g in result.graded_answers}
    assert graded_by_q[2].is_correct is None
    assert graded_by_q[3].is_correct is None


def test_missing_detection_counts_as_unanswered():
    detected = [DetectedAnswer(1, "A", AnswerState.CLEAR)]
    result = grade_submission(detected, _key())
    assert result.unanswered_count == 2
    assert result.correct_count == 1
