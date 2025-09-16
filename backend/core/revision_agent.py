from typing import List, Dict, Any
from .llm import GeminiLLMWrapper
from backend.prompts import revision_prompts
import asyncio

llm_wrapper = GeminiLLMWrapper()

class RevisionAgent:
    def __init__(self, llm=None):
        self.llm = llm or llm_wrapper

    async def generate_explanation_steps(self, title: str, content: str, conversation_history: str = "", steps: int = 3) -> List[str]:
        prompt = revision_prompts.EXPLANATION_TEMPLATE.format(
            title=title, steps=steps, content=content, conversation_history=conversation_history
        )
        # call LLM
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        # naive parse: split into lines, take the top `steps` non-empty lines
        lines = [l.strip() for l in resp.splitlines() if l.strip()]
        if len(lines) == 0:
            # fallback: split sentences from response
            parts = [s.strip() for s in resp.replace("\n", " ").split(".") if s.strip()]
            lines = [f"{i+1}. {parts[i]}." for i in range(min(steps, len(parts)))]
        return lines[:steps]

    async def make_check_question(self, title: str, content: str, conversation_history: str = "") -> str:
        prompt = revision_prompts.CHECK_QUESTION_TEMPLATE.format(
            title=title, content=content, conversation_history=conversation_history
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        return resp.strip()

    async def extract_expected_keywords(self, title: str, content: str, question: str) -> List[str]:
        """Ask LLM to return minimal keywords for marking correctness."""
        prompt = revision_prompts.KEYWORDS_EXTRACTION_TEMPLATE.format(
            title=title,
            content=content,
            question=question
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        text = resp.strip()
        try:
            import json
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x).strip().lower() for x in data if str(x).strip()]
        except Exception:
            pass
        # fallback: use title tokens
        return [w.lower() for w in title.split()[:3]]

    async def evaluate_answer(self, user_answer: str, expected_keywords: List[str], conversation_history: str = "", *, title: str = "", content: str = "", assistant_message: str = "", check_question: str = "") -> Dict[str, Any]:
        """Evaluate using full context instead of relying on keyword matches."""
        # Prefer full-context evaluation
        prompt = revision_prompts.EVAL_WITH_CONTEXT_TEMPLATE.format(
            title=title,
            content=content,
            assistant_message=assistant_message,
            check_question=check_question,
            user_answer=user_answer
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        # Parse structured verdict
        verdict = "WRONG"
        justification = ""
        correction = ""
        for line in resp.splitlines():
            line_low = line.strip()
            if line_low.upper().startswith("VERDICT:"):
                verdict = line_low.split(":",1)[1].strip().upper()
            elif line_low.upper().startswith("JUSTIFICATION:"):
                justification = line_low.split(":",1)[1].strip()
            elif line_low.upper().startswith("CORRECTION:"):
                correction = line_low.split(":",1)[1].strip()
        return {"verdict": verdict, "justification": justification, "correction": correction or justification}

    async def handle_qa_request(self, user_question: str, current_concept: str, content: str, conversation_history: str = "") -> str:
        """Handle Q&A requests during revision sessions"""
        prompt = revision_prompts.QA_RESPONSE_TEMPLATE.format(
            user_question=user_question,
            current_concept=current_concept,
            content=content,
            conversation_history=conversation_history
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        return resp.strip()

    async def detect_question_intent(self, user_input: str, current_concept: str, conversation_history: str = "") -> str:
        """Dynamically detect if user input is a question or an answer using LLM"""
        prompt = revision_prompts.QUESTION_DETECTION_TEMPLATE.format(
            user_input=user_input,
            current_concept=current_concept,
            conversation_history=conversation_history
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        # Clean up the response to get just the classification
        classification = resp.strip().upper()
        if "ASKING_QUESTION" in classification:
            return "ASKING_QUESTION"
        elif "PROVIDING_ANSWER" in classification:
            return "PROVIDING_ANSWER"
        else:
            # Fallback: if LLM response is unclear, default to treating as answer
            return "PROVIDING_ANSWER"