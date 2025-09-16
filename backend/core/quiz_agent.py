# backend/core/quiz_agent.py
from typing import List, Dict, Any
from .llm import GeminiLLMWrapper
from backend.prompts import quiz_prompts

llm_wrapper = GeminiLLMWrapper()

class QuizAgent:
    def __init__(self, llm=None):
        self.llm = llm or llm_wrapper

    async def generate_quiz(self, title: str, content: str, conversation_history: str = "", n: int = 3) -> List[Dict[str, Any]]:
        prompt = quiz_prompts.QUIZ_GENERATION_TEMPLATE.format(n=n, title=title, content=content, conversation_history=conversation_history)
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        # We expect JSON-like results; but to keep robust we'll return the raw text inside a list.
        return [{"raw": resp}]

    async def evaluate_quiz_answer(self, user_answer: str, correct_answer: str, conversation_history: str = "") -> Dict[str, Any]:
        prompt = quiz_prompts.QUIZ_EVAL_TEMPLATE.format(user_answer=user_answer, correct=correct_answer, conversation_history=conversation_history)
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        return {"llm_response": resp}
