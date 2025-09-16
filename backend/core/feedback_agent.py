from backend.prompts import feedback_prompts

class FeedbackAgent:
    def __init__(self):
        pass

    def feedback_for(self, verdict: str, details: dict) -> str:
        if verdict == "CORRECT":
            return feedback_prompts.FEEDBACK_CORRECT
        if verdict == "PARTIAL":
            correction = details.get("correction", "Add missing parts.")
            return feedback_prompts.FEEDBACK_PARTIAL.format(correction=correction)
        correction = details.get("correction", "Let's review this concept again.")
        return feedback_prompts.FEEDBACK_WRONG.format(correction=correction)
