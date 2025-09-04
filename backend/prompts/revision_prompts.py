from typing import List
class RevisionPrompts:
    """
    Enhanced prompts for LangGraph-based step-by-step learning flow.
    Supports short, focused interactions with immediate feedback.
    """
    
    @staticmethod
    def get_topic_introduction_prompt(topic: str, subtopics_count: int, last_bot_message: str = None) -> str:
        """Brief topic overview + readiness check"""
        context = f"Previous message: '{(last_bot_message or '').strip()}'\n" if last_bot_message else ""
        
        return f"""
        {context}
        Introduce the topic "{topic}" which has {subtopics_count} concepts to learn.
        
        REQUIREMENTS:
        - 1 sentence overview of what this topic covers
        - Mention we'll learn step-by-step with quick checks
        - End with: "Ready to start with the first concept?"
        - Be encouraging and use 1 emoji
        - Keep to 2-3 lines total
        """
    
    @staticmethod
    def get_step_by_step_explanation_prompt(subtopic_title: str, subtopic_content: str, step: int, total_steps: int, last_bot_message: str = None) -> str:
        """Step-by-step explanation broken into small messages"""
        context = f"Previous message: '{(last_bot_message or '').strip()}'\n" if last_bot_message else ""
        
        step_focus = {
            1: "Start with the basic definition and what this concept is about",
            2: "Explain how it works or the main process/mechanism", 
            3: "Give a simple example or real-world application"
        }
        
        focus = step_focus.get(step, "Continue explaining the concept")
        
        return f"""
        {context}
        You are explaining "{subtopic_title}" step by step.
        
        Content: {subtopic_content[:400]}
        
        This is Message {step} of {total_steps}.
        Focus for this message: {focus}
        
        REQUIREMENTS:
        - Explain just this step in 1-2 simple sentences
        - Use easy language a student can understand
        - Be clear and engaging
        - Don't ask questions yet - just explain
        - If step {total_steps}, add "Got that so far?"
        """
    
    @staticmethod
    def get_simple_check_question_prompt(subtopic_title: str, subtopic_content: str, last_bot_message: str = None) -> str:
        """Generate a simple check question after explanation"""
        context = f"Previous message: '{(last_bot_message or '').strip()}'\n" if last_bot_message else ""
        
        return f"""
        {context}
        Create ONE simple question to check if the student understood "{subtopic_title}".
        
        Content: {subtopic_content[:300]}
        
        REQUIREMENTS:
        - Ask ONE clear, simple question
        - Make it easy to answer (not too complex)
        - Can be multiple choice, true/false, or short answer
        - Focus on the main concept only
        - Keep the question to 1 line
        - Don't explain - just ask the question
        """
    
    @staticmethod
    def get_answer_feedback_prompt(user_answer: str, is_correct: bool, concept_name: str, last_bot_message: str = None) -> str:
        """Give feedback on student's answer"""
        context = f"Previous message: '{(last_bot_message or '').strip()}'\n" if last_bot_message else ""
        
        if is_correct:
            return f"""
            {context}
            The student answered correctly about "{concept_name}".
            Student's answer: "{user_answer}"
            
            REQUIREMENTS:
            - Give positive feedback (1 line)
            - Briefly confirm what they got right
            - Say "Great! Let's move to the next concept."
            - Be encouraging and use 1 positive emoji
            - Keep to 2 lines total
            """
        else:
            return f"""
            {context}
            The student's answer about "{concept_name}" needs improvement.
            Student's answer: "{user_answer}"
            
            REQUIREMENTS:
            - Be gentle and encouraging (1 line)
            - Give a quick hint or correction
            - Say "No worries! Let's continue to the next concept."
            - Keep positive tone with encouraging emoji
            - Keep to 2 lines total
            """
    
    @staticmethod
    def get_user_question_response_prompt(user_question: str, content_context: str, last_bot_message: str = None) -> str:
        """Respond to user's question during learning"""
        context = f"Previous message: '{(last_bot_message or '').strip()}'\n" if last_bot_message else ""
        
        return f"""
        {context}
        Student asked: "{user_question}"
        
        Context content: {content_context}
        
        REQUIREMENTS:
        - Answer their question briefly and clearly (1-2 lines)
        - Use the content context to give accurate info
        - Be helpful and encouraging
        - End with "Does that help? Ready to continue learning?"
        - Keep response short and focused
        """
    
    @staticmethod
    def get_session_summary_prompt(total_concepts: int, concepts_learned: int, avg_score: float, learned_concepts: List[str], last_bot_message: str = None) -> str:
        """Final session summary"""
        context = f"Previous message: '{(last_bot_message or '').strip()}'\n" if last_bot_message else ""
        
        performance = "excellent" if avg_score >= 0.8 else "good" if avg_score >= 0.6 else "needs practice"
        
        return f"""
        {context}
        Session complete! Student learned {concepts_learned} out of {total_concepts} concepts.
        Average performance: {avg_score:.1%} ({performance})
        Concepts covered: {', '.join(learned_concepts[:3])}{'...' if len(learned_concepts) > 3 else ''}
        
        REQUIREMENTS:
        - Celebrate their learning journey (1 line)
        - Mention specific numbers (concepts learned, performance)
        - Give encouraging words about their progress
        - End with motivational message and celebration emoji
        - Keep to 3-4 lines total
        - Be warm and positive
        """
    
    @staticmethod
    def get_transition_to_next_concept_prompt(next_concept_title: str, current_progress: str, last_bot_message: str = None) -> str:
        """Smooth transition between concepts"""
        context = f"Previous message: '{(last_bot_message or '').strip()}'\n" if last_bot_message else ""
        
        return f"""
        {context}
        Moving to the next concept: "{next_concept_title}"
        Current progress: {current_progress}
        
        REQUIREMENTS:
        - Brief transition (1 line): "Now let's learn about [concept]"
        - Keep it smooth and encouraging
        - Don't explain yet - just introduce what's coming
        - Use transition words like "Next" or "Now"
        - Add one forward-looking emoji
        """
