import json
from pathlib import Path

import pytest


DATASET = (
    Path(__file__).resolve().parent.parent
    / "eval_datasets"
    / "state_questions.json"
)

with open(DATASET, encoding="utf-8") as file:
    test_cases = json.load(file)


@pytest.mark.parametrize(
    "case",
    test_cases,
    ids=[case["id"] for case in test_cases]
)
def test_answer_correctness(chat, case):
    result = chat(case["question"])

    answer = result["response"].lower()

    for keyword in case["expected_keywords"]:
        assert keyword.lower() in answer, (
            f"{case['id']}: Expected '{keyword}' "
            f"in response: {result['response']}"
        )