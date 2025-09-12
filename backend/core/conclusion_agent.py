from .llm import GeminiLLMWrapper
from backend.prompts import conclusion_prompts

llm_wrapper = GeminiLLMWrapper()

class ConclusionAgent:
    def __init__(self, llm=None):
        self.llm = llm or llm_wrapper

    async def summary(self, correct: int, total: int, conversation_history: str = "") -> str:
        prompt = conclusion_prompts.CONCLUSION_TEMPLATE.format(correct=correct, total=total, conversation_history=conversation_history)
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        return resp.strip()
