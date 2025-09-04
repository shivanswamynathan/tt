class RevisionPrompts:
    """
    Mobile-optimized prompts for subtopic-based learning flow.
    All responses limited to 1-2 lines maximum.
    """
    
    @staticmethod
    def get_topic_introduction_prompt(topic: str, subtopics_count: int, last_bot_message: str = None) -> str:
        """Brief topic overview + proceed question"""
        context = f"Previous assistant message: '{(last_bot_message or '').strip()}'\n" if last_bot_message else ""
        
        return f"""
        {context}
        Give a 1-sentence overview of "{topic}" which has {subtopics_count} subtopics.
        Then ask: "Ready to start with the first concept?" 
        
        - Exactly 4-5 lines (more interactive)

        Be engaging and use 1 emoji.
        """
    
    @staticmethod
    def get_subtopic_learning_prompt(subtopic_title: str, subtopic_content: str, subtopic_number: str, last_bot_message: str = None) -> str:
        """Mini lesson for one subtopic"""
        context = f"Previous assistant message (for continuity): '{(last_bot_message or '').strip()}'\n" if last_bot_message else ""
        
        return f"""
        {context}
        Explain subtopic {subtopic_number}: "{subtopic_title}" in 4-5 lines.
        
        Content: {subtopic_content[:300]}
        
        REQUIREMENTS:
        - Exactly 4-5 lines (more interactive)
        - Simple language with examples
        - Make it engaging and conversational
        - End with "Got it? Ready for a quick check?"
        """
    
    @staticmethod
    def get_subtopic_quiz_prompt(subtopic_title: str, subtopic_content: str, subtopic_number: str, last_bot_message: str = None) -> str:
        """2-question quiz per subtopic"""
        context = f"Previous assistant message (for context): '{(last_bot_message or '').strip()}'\n" if last_bot_message else ""
        
        return f"""
        {context}
        Create  2 questions for subtopic {subtopic_number}: "{subtopic_title}"
        
        Content: {subtopic_content}
        
        FORMAT:
        Q1: [Simple MCQ with A,B,C options]
        Q2: [True/False question]
        
        Keep questions short and focused on key concept only.
        """
    
    @staticmethod
    def get_subtopic_feedback_prompt(user_answers: str, correct_answers: str, passed: bool, subtopic_number: str, last_bot_message: str = None) -> str:
        """Detailed pass/fail feedback for subtopic"""
        context = f"Previous assistant message (for context): '{(last_bot_message or '').strip()}'\n" if last_bot_message else ""
        
        if passed:
            return f"""
            {context}
            Student passed subtopic {subtopic_number} quiz.
            Student answers: {user_answers}
            
            REQUIREMENTS:
            - Check each answer: say "Q1: Correct!" or "Q1: Wrong, correct answer is..."
            - Give brief explanation for any wrong answers
            - Celebrate success in 4-5 lines total
            - End with "Ready to move to the next concept?"
            """
        else:
            return f"""
            {context}
            Student failed subtopic {subtopic_number} quiz.
            Student answers: {user_answers}
            
            REQUIREMENTS:
            - Check each answer: say "Q1: Correct!" or "Q1: Wrong, correct answer is..."
            - Give clear explanations for wrong answers
            - Be encouraging in 4-5 lines total
            - End with "Want me to explain this concept again?"
            """
    
    @staticmethod
    def get_final_assessment_prompt(overall_score: float, total_subtopics: int, passed_subtopics: int, last_bot_message: str = None) -> str:
        """Final session completion summary"""
        context = f"Previous assistant message (for context): '{(last_bot_message or '').strip()}'\n" if last_bot_message else ""
        
        return f"""
        {context}
        Student completed all {total_subtopics} subtopics with {passed_subtopics} passed.
        Overall score: {overall_score:.1%}
        
        REQUIREMENTS:
        - Celebrate their achievement in 4-5 lines
        - Mention specific accomplishments (subtopics passed, score)
        - Give encouraging final words about their learning
        - End with celebration emoji
        - Be warm and motivating
        """

    @staticmethod  
    def get_retry_explanation_prompt(subtopic_title: str, subtopic_content: str, subtopic_number: str, last_bot_message: str = None) -> str:
        """Simpler re-explanation for failed subtopic"""
        context = f"Previous assistant message (for context): '{(last_bot_message or '').strip()}'\n" if last_bot_message else ""
        
        return f"""
        {context}
        Re-explain subtopic {subtopic_number}: "{subtopic_title}" more simply.
        
        Content: {subtopic_content[:200]}
        
        REQUIREMENTS:
        - Explain in 4-5 lines with simpler words
        - Use easy, relatable examples
        - Break down complex ideas step by step
        - Be encouraging and supportive
        - End with "Clearer now? Let's try the quiz again!"
        """