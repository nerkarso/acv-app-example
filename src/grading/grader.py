"""Deterministic grading engine. No AI or OCR dependency -- pure function
over detected answers (state='clear' only) and an answer key."""

from __future__ import annotations

from src.schemas import AnswerState, GradedAnswer, GradingResult


class DetectedAnswer:
    def __init__(self, question_number: int, detected_answer: str | None, answer_state: AnswerState):
        self.question_number = question_number
        self.detected_answer = detected_answer
        self.answer_state = answer_state


class AnswerKeyEntry:
    def __init__(self, question_number: int, correct_answer: str, points: float = 1.0):
        self.question_number = question_number
        self.correct_answer = correct_answer
        self.points = points


def grade_submission(
    detected_answers: list[DetectedAnswer],
    answer_key: list[AnswerKeyEntry],
    submission_id: int | None = None,
) -> GradingResult:
    key_by_number = {entry.question_number: entry for entry in answer_key}
    detected_by_number = {a.question_number: a for a in detected_answers}

    graded: list[GradedAnswer] = []
    correct_count = 0
    incorrect_count = 0
    unanswered_count = 0
    score = 0.0
    total_points = sum(entry.points for entry in answer_key)

    for question_number, key_entry in sorted(key_by_number.items()):
        detected = detected_by_number.get(question_number)

        if detected is None or detected.answer_state != AnswerState.CLEAR:
            state = detected.answer_state if detected is not None else AnswerState.BLANK
            graded.append(
                GradedAnswer(
                    question_number=question_number,
                    detected_answer=detected.detected_answer if detected else None,
                    correct_answer=key_entry.correct_answer,
                    is_correct=None,
                    answer_state=state,
                )
            )
            unanswered_count += 1
            continue

        is_correct = detected.detected_answer == key_entry.correct_answer
        if is_correct:
            correct_count += 1
            score += key_entry.points
        else:
            incorrect_count += 1

        graded.append(
            GradedAnswer(
                question_number=question_number,
                detected_answer=detected.detected_answer,
                correct_answer=key_entry.correct_answer,
                is_correct=is_correct,
                answer_state=AnswerState.CLEAR,
            )
        )

    percentage = (score / total_points * 100.0) if total_points > 0 else 0.0

    return GradingResult(
        submission_id=submission_id,
        graded_answers=graded,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        unanswered_count=unanswered_count,
        score=score,
        total_points=total_points,
        percentage=percentage,
    )
