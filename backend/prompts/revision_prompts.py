
class RevisionPrompts:
    """
    Template generator for AI tutoring prompts in revision system.
    - Provides structured prompts for teaching, quizzing, and feedback phases
    - Ensures consistent AI response format and educational quality
    - Adapts content based on learning context and student performance
    """
    
    @staticmethod
    def get_basic_revision_prompt(topic: str, content: str, user_query: str = None, is_start: bool = False) -> str:
        """Type 1: Basic Revision Helper - Main teaching and question answering"""
        
        if is_start:
            return f"""
            Start a revision session for "{topic}". 
            
            REQUIREMENTS:
            - Give a friendly welcome
            - Provide a concise overview (max 250 words) 
            - Explain key concepts clearly with examples
            - End with "Ready for a quick quiz to test your understanding? 🧠"
            - Use engaging language and emojis
            
            Content about the topic:
            {content}
            
            Create an engaging start to the revision session.
            """
        
        elif user_query:
            return f"""
            The student asked a question about "{topic}".
            
            Student's question: "{user_query}"
            
            REQUIREMENTS:
            - Answer their question clearly and completely
            - Use the content below to provide accurate information
            - Give practical examples they can understand
            - Be encouraging and supportive
            - Ask if they have more questions or want to continue learning
            
            Relevant content:
            {content}
            
            Answer their question helpfully.
            """
        
        else:
            return f"""
            Continue teaching about "{topic}".
            
            REQUIREMENTS:
            - Explain the next concept clearly
            - Use simple language and good examples
            - Keep it engaging and interactive
            - End by asking if they're ready for a quiz or have questions
            
            Content to teach:
            {content}
            
            Continue the revision lesson.
            """
    
    @staticmethod
    def get_auto_quiz_prompt(topic: str, concepts: list, difficulty: str = "medium") -> str:
        """Type 2: Auto Quiz Generator - Creates quizzes automatically"""
        concepts_text = ", ".join(concepts) if concepts else topic
        
        return f"""
        Create an automatic quiz for "{topic}" covering: {concepts_text}
        
        REQUIREMENTS:
        - Create exactly 3 questions (mix of MCQ, True/False, Fill-in-blank)
        - Difficulty level: {difficulty}
        - Clear numbering: Q1, Q2, Q3
        - For MCQ: provide options A, B, C
        - Make questions test real understanding, not just memory
        - End with "Take your time and answer all three questions! 📝"
        
        EXACT FORMAT:
        🧠 **Quick Quiz Time!**
        
        **Q1:** [Multiple choice question]
        A) [Option 1]
        B) [Option 2]  
        C) [Option 3]
        
        **Q2:** True or False: [Statement]
        
        **Q3:** Fill in the blank: [Sentence with ___]
        
        Take your time and answer all three questions! 📝
        
        Generate the quiz following this exact format.
        """
    
    @staticmethod
    def get_feedback_progress_prompt(user_answers: str, topic: str, correct_answers: str = "", performance_level: str = "good") -> str:
        """Type 3: Feedback & Progress - Quiz feedback and learning progress"""
        
        if performance_level == "poor":  # ≤50% performance
            return f"""
            The student performed poorly (≤50%) on the quiz about "{topic}".
            
            Student's answers: {user_answers}
            Correct answers: {correct_answers}
            
            REQUIREMENTS:
            - Be very encouraging and supportive (NOT discouraging)
            - Explain the correct answers with simple examples
            - Provide helpful hints and analogies
            - Offer to explain concepts more simply
            - End with "Would you like me to explain these concepts more simply? I'm here to help! 🤗"
            - Use supportive emojis
            
            Provide encouraging feedback and offer remedial help.
            """
        
        elif performance_level == "good":  # >50% performance
            return f"""
            The student performed well (>50%) on the quiz about "{topic}".
            
            Student's answers: {user_answers}
            Correct answers: {correct_answers}
            
            REQUIREMENTS:
            - Celebrate their success enthusiastically
            - Mention specific things they got right
            - Briefly explain any missed concepts
            - Show their learning progress
            - Suggest continuing to next concepts or deeper exploration
            - End with asking what they want to learn next
            - Use celebratory emojis
            
            Provide encouraging feedback and suggest next steps.
            """
        
        else:  # General progress update
            return f"""
            Provide a progress update for the student learning "{topic}".
            
            REQUIREMENTS:
            - Show what they've learned so far
            - Celebrate their effort and progress
            - Suggest what to explore next
            - Keep it motivating and positive
            - Ask what they want to focus on next
            
            Create a motivating progress update.
            """