from .llm import GeminiLLMWrapper
from backend.prompts import qa_prompts

llm_wrapper = GeminiLLMWrapper()

class QAAgent:
    def __init__(self, llm=None):
        self.llm = llm or llm_wrapper

    async def answer_question(self, question: str, conversation_history: str = "", content: str = "") -> str:
        prompt = qa_prompts.QA_ANSWER_TEMPLATE.format(question=question, conversation_history=conversation_history, content=content)
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        return resp.strip()
