
QUIZ_GENERATION_TEMPLATE = """
Generate {n} short questions (multiple-choice or short answer) that test the concept '{title}'.
Each question should be simple and linked to the concept content. Return as a JSON-like list (question, options if any, correct_answer).
Include conversation history for context:
{conversation_history}
Content:
{content}
"""

QUIZ_EVAL_TEMPLATE = """
User answer: {user_answer}
Correct answer: {correct}
Conversation history:
{conversation_history}

Return: VERDICT: <CORRECT|PARTIAL|WRONG>\\nFEEDBACK: <one short sentence>
"""
