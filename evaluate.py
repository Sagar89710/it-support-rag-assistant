"""Run a small offline evaluation set for the support assistant."""
from app import answer_question


EVALUATION_CASES = [
    ("How do I reset my password?", "Password Reset"),
    ("How do I connect to the corporate VPN?", "VPN Access"),
    ("What should I include when escalating an incident?", "Incident Escalation"),
]


def main() -> None:
    passed = 0
    print("Enterprise IT Support Assistant Evaluation")
    print("=" * 46)

    for question, expected_section in EVALUATION_CASES:
        result = answer_question(question)
        source_text = " ".join(result.get("sources", []))
        is_pass = expected_section in source_text and result.get("validation") == "passed"
        passed += int(is_pass)
        status = "PASS" if is_pass else "FAIL"
        print(f"[{status}] {expected_section}: {result.get('validation')}")

    print(f"\nRetrieval accuracy: {passed}/{len(EVALUATION_CASES)}")


if __name__ == "__main__":
    main()
