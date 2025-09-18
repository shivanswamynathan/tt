import streamlit as st
import requests
import json
from datetime import datetime
import uuid

# Page configuration
st.set_page_config(
    page_title="EduBot - Progressive Revision",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000/api"

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None
if "student_id" not in st.session_state:
    st.session_state.student_id = "student_001"
if "conversation_count" not in st.session_state:
    st.session_state.conversation_count = 0
if "revision_messages" not in st.session_state:
    st.session_state.revision_messages = []
if "session_complete" not in st.session_state:
    st.session_complete = False

def main():
    st.title("📚 EduBot - Progressive Topic Revision")
    
    # Sidebar for topic selection
    with st.sidebar:
        st.header("📖 Available Topics")
        
        # Fetch available topics
        topics = fetch_available_topics()
        
        if topics:
            topic_options = {topic["topic"]: f"{topic['topic']} ({topic['chunk_count']} sections)" 
                           for topic in topics}
            
            selected_topic = st.selectbox(
                "Choose a topic to revise:",
                options=list(topic_options.keys()),
                format_func=lambda x: topic_options[x],
                key="topic_selector"
            )
            
            # Topic details
            if selected_topic:
                topic_info = next(t for t in topics if t["topic"] == selected_topic)
                st.info(f"📋 {topic_info['description']}")
            
            # Start new session button
            if st.button("🚀 Start New Revision Session", type="primary"):
                if selected_topic:
                    start_new_session(selected_topic)
                else:
                    st.error("Please select a topic first")
            
            # Session info
            if st.session_state.session_id:
                st.header("📊 Current Session")
                st.write(f"**Topic:** {st.session_state.current_topic}")
                st.write(f"**Progress:** {st.session_state.conversation_count}/20 interactions")
                
                progress = min(st.session_state.conversation_count / 20, 1.0)
                st.progress(progress)
                
                if st.session_state.conversation_count > 0:
                    if st.session_state.conversation_count <= 5:
                        stage = "🌱 Introduction Stage"
                    elif st.session_state.conversation_count <= 15:
                        stage = "🧠 Deep Learning Stage"
                    else:
                        stage = "🎯 Consolidation Stage"
                    
                    st.write(f"**Current Stage:** {stage}")
                
                # End session button
                if st.button("🏁 End Session Early"):
                    end_session()
        else:
            st.error("Could not fetch topics from database")
    
    # Main content area
    if not st.session_state.session_id:
        show_welcome_screen()
    else:
        show_revision_interface()

def show_welcome_screen():
    """Show welcome screen when no session is active"""
    st.header("Welcome to Progressive Topic Revision! 🎓")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        ### How Progressive Revision Works:
        
        **🌱 Introduction Stage (1-5 interactions)**
        - Learn fundamental concepts and definitions
        - Build understanding with simple explanations
        - Answer basic comprehension questions
        
        **🧠 Deep Learning Stage (6-15 interactions)**
        - Explore detailed concepts and relationships
        - Apply knowledge to real-world scenarios
        - Tackle analytical and critical thinking questions
        
        **🎯 Consolidation Stage (16-20 interactions)**
        - Synthesize and summarize key learnings
        - Test comprehensive understanding
        - Prepare for assessments
        
        **💡 Interactive Features:**
        - Ask questions anytime during revision
        - Get personalized explanations
        - Receive encouraging feedback
        - Track your progress throughout the session
        """)
    
    with col2:
        st.info("""
        **📚 Getting Started:**
        
        1. Select a topic from the sidebar
        2. Click "Start New Revision Session"
        3. Follow the interactive guidance
        4. Ask questions freely
        5. Complete the full session for best results
        
        **💬 Tips for Success:**
        - Engage actively with questions
        - Don't hesitate to ask for clarification
        - Take your time to understand concepts
        - Participate in all stages for maximum benefit
        """)

def show_revision_interface():
    """Show the main revision chat interface - Updated for unlimited conversations"""
    st.header(f"📖 Revising: {st.session_state.current_topic}")
    
    # Show current stage and conversation count (no limits!)
    if st.session_state.conversation_count > 0:
        stage_info = {
            "introduction": "🌱 Introduction Stage",
            "deep_learning": "🧠 Deep Learning Stage", 
            "consolidation": "🎯 Consolidation Stage",
            "advanced_exploration": "🚀 Advanced Exploration",
            "mastery_discussion": "🎓 Mastery Discussion"
        }
        
        # Determine current stage
        count = st.session_state.conversation_count
        if count <= 5:
            current_stage = "🌱 Introduction Stage"
        elif count <= 15:
            current_stage = "🧠 Deep Learning Stage"
        elif count <= 25:
            current_stage = "🎯 Consolidation Stage"
        elif count <= 40:
            current_stage = "🚀 Advanced Exploration"
        else:
            current_stage = "🎓 Mastery Discussion"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Conversations", st.session_state.conversation_count)
        with col2:
            st.info(f"**Stage:** {current_stage}")
        with col3:
            if st.button("🏁 End Session"):
                handle_user_input("end session")
    
    # Display revision messages
    for message in st.session_state.revision_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                
                # Show current stage info
                if metadata.get('current_stage'):
                    stage_names = {
                        "introduction": "🌱 Introduction",
                        "deep_learning": "🧠 Deep Learning", 
                        "consolidation": "🎯 Consolidation",
                        "advanced_exploration": "🚀 Advanced",
                        "mastery_discussion": "🎓 Mastery"
                    }
                    stage_display = stage_names.get(metadata['current_stage'], metadata['current_stage'])
                    st.caption(f"Stage: {stage_display} | Interaction #{metadata.get('conversation_count', 0)}")
                
                # Show sources if available
                if metadata.get('sources'):
                    with st.expander("📚 Content Sources"):
                        for source in metadata['sources']:
                            st.write(f"• Section {source}")
    
    # Chat input - now always available unless session is manually completed
    if not st.session_state.session_complete:
        st.info("💡 **Unlimited Learning**: Continue as long as you want! Type 'end session' when you're ready to finish.")
        if prompt := st.chat_input("Ask a question, continue learning, or type 'end session' to finish..."):
            handle_user_input(prompt)
    else:
        st.success("🎉 Session completed! Great job on your learning journey!")
        if st.button("🚀 Start New Session"):
            # Reset for new session
            st.session_state.session_complete = False
            st.rerun()

def fetch_available_topics():
    """Fetch available topics from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/topics")
        if response.status_code == 200:
            return response.json()["topics"]
        else:
            st.error(f"Failed to fetch topics: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

def start_new_session(topic):
    """Start a new revision session"""
    try:
        # Reset session state
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.current_topic = topic
        st.session_state.conversation_count = 0
        st.session_state.revision_messages = []
        st.session_state.session_complete = False
        
        # Call API to start session
        data = {
            "topic": topic,
            "student_id": st.session_state.student_id,
            "session_id": st.session_state.session_id,
            "conversation_count": 0
        }
        
        response = requests.post(f"{API_BASE_URL}/revision/start", json=data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Add assistant's welcome message
            assistant_message = {
                "role": "assistant",
                "content": result["response"],
                "metadata": {
                    "conversation_count": result["conversation_count"],
                    "sources": result.get("sources", []),
                    "is_session_complete": result["is_session_complete"]
                }
            }
            
            st.session_state.revision_messages.append(assistant_message)
            st.session_state.conversation_count = result["conversation_count"]
            
            st.success(f"✅ Started revision session for '{topic}'!")
            st.rerun()
        else:
            st.error(f"Failed to start session: {response.json().get('detail', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"Error starting session: {str(e)}")

def handle_user_input(user_input):
    """Handle user input and get response from revision agent"""
    try:
        # Add user message to chat
        user_message = {"role": "user", "content": user_input}
        st.session_state.revision_messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Call API to continue session
        data = {
            "topic": st.session_state.current_topic,
            "query": user_input,
            "session_id": st.session_state.session_id,
            "student_id": st.session_state.student_id,
            "conversation_count": st.session_state.conversation_count
        }
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(f"{API_BASE_URL}/revision/continue", json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.write(result["response"])
                    
                    # Update session state
                    st.session_state.conversation_count = result["conversation_count"]
                    st.session_state.session_complete = result["is_session_complete"]
                    
                    # Show progress and metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Progress", f"{result['conversation_count']}/20")
                    with col2:
                        if result.get('sources'):
                            st.metric("Sources", len(result['sources']))
                    with col3:
                        if result['is_session_complete']:
                            st.success("✅ Session Complete!")
                    
                    # Show sources
                    if result.get('sources'):
                        with st.expander("📚 Content Sources"):
                            for source in result['sources']:
                                st.write(f"• Section {source}")
                    
                    # Show session summary if complete
                    if result['is_session_complete'] and result.get('session_summary'):
                        st.info("🎓 **Session Summary**")
                        st.write(result['session_summary'])
                        
                        if result.get('next_suggested_action'):
                            st.info(f"**Next Steps:** {result['next_suggested_action']}")
                    
                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant",
                        "content": result["response"],
                        "metadata": {
                            "conversation_count": result["conversation_count"],
                            "sources": result.get("sources", []),
                            "is_session_complete": result["is_session_complete"],
                            "session_summary": result.get("session_summary"),
                            "next_suggested_action": result.get("next_suggested_action")
                        }
                    }
                    
                    st.session_state.revision_messages.append(assistant_message)
                    
                else:
                    error_msg = f"Error: {response.json().get('detail', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.revision_messages.append({"role": "assistant", "content": error_msg})
    
    except Exception as e:
        error_msg = f"Connection error: {str(e)}"
        st.error(error_msg)
        st.session_state.revision_messages.append({"role": "assistant", "content": error_msg})

def end_session():
    """End the current revision session"""
    st.session_state.session_id = None
    st.session_state.current_topic = None
    st.session_state.conversation_count = 0
    st.session_state.revision_messages = []
    st.session_state.session_complete = False
    st.success("Session ended. You can start a new revision session anytime!")
    st.rerun()

if __name__ == "__main__":
    main()

# frontend/pages/topic_overview.py
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def topic_overview_page():
    """Page showing overview of all available topics"""
    st.title("📊 Topic Overview & Analytics")
    
    # Fetch topics
    topics = fetch_available_topics()
    
    if not topics:
        st.error("Could not load topics from database")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Topics", len(topics))
    
    with col2:
        total_chunks = sum(topic.get("chunk_count", 0) for topic in topics)
        st.metric("Total Content Sections", total_chunks)
    
    with col3:
        avg_chunks = total_chunks / len(topics) if topics else 0
        st.metric("Avg Sections per Topic", f"{avg_chunks:.1f}")
    
    with col4:
        # This would come from session tracking in a real app
        st.metric("Topics Available", len(topics))
    
    # Topic content distribution chart
    st.header("📈 Content Distribution by Topic")
    
    if topics:
        # Create bar chart
        fig = px.bar(
            x=[topic["topic"] for topic in topics],
            y=[topic.get("chunk_count", 0) for topic in topics],
            title="Number of Content Sections per Topic",
            labels={"x": "Topics", "y": "Content Sections"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Topic details table
    st.header("📚 Topic Details")
    
    for topic in topics:
        with st.expander(f"📖 {topic['topic']} ({topic.get('chunk_count', 0)} sections)"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Description:** {topic.get('description', 'No description available')}")
                st.write(f"**Content Sections:** {topic.get('chunk_count', 0)}")
                
                # Sample content preview (would fetch from MongoDB)
                if st.button(f"Preview Content", key=f"preview_{topic['topic']}"):
                    preview_topic_content(topic['topic'])
            
            with col2:
                if st.button(f"Start Revision", key=f"start_{topic['topic']}", type="primary"):
                    st.session_state.selected_topic_for_revision = topic['topic']
                    st.switch_page("app.py")

def preview_topic_content(topic_name):
    """Show a preview of topic content"""
    try:
        # This would call an API endpoint to get sample content
        st.info(f"Loading preview for {topic_name}...")
        
        # Placeholder for actual content preview
        st.write("**Sample Content Preview:**")
        st.write("*This is where a preview of the topic content would appear...*")
        
    except Exception as e:
        st.error(f"Error loading preview: {str(e)}")

def fetch_available_topics():
    """Fetch available topics from the API"""
    try:
        response = requests.get("http://localhost:8000/api/topics")
        if response.status_code == 200:
            return response.json()["topics"]
        else:
            return []
    except Exception as e:
        st.error(f"Error fetching topics: {str(e)}")
        return []

# frontend/pages/session_history.py
import streamlit as st
from datetime import datetime, timedelta
import json

def session_history_page():
    """Page showing revision session history"""
    st.title("📈 Your Revision History")
    
    # Initialize session history if not exists
    if "session_history" not in st.session_state:
        st.session_state.session_history = []
    
    if not st.session_state.session_history:
        st.info("🔍 No revision sessions completed yet. Start your first session!")
        if st.button("🚀 Start First Revision Session"):
            st.switch_page("app.py")
        return
    
    # Display session statistics
    st.header("📊 Session Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_sessions = len(st.session_state.session_history)
    completed_sessions = sum(1 for s in st.session_state.session_history if s.get("completed", False))
    total_interactions = sum(s.get("interaction_count", 0) for s in st.session_state.session_history)
    unique_topics = len(set(s.get("topic", "") for s in st.session_state.session_history))
    
    with col1:
        st.metric("Total Sessions", total_sessions)
    with col2:
        st.metric("Completed Sessions", completed_sessions)
    with col3:
        st.metric("Total Interactions", total_interactions)
    with col4:
        st.metric("Topics Studied", unique_topics)
    
    # Session list
    st.header("📚 Recent Sessions")
    
    for i, session in enumerate(reversed(st.session_state.session_history)):
        with st.expander(
            f"📖 {session.get('topic', 'Unknown Topic')} - "
            f"{session.get('date', 'Unknown Date')} "
            f"({'✅ Completed' if session.get('completed', False) else '⏸️ Incomplete'})"
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Topic:** {session.get('topic', 'Unknown')}")
                st.write(f"**Date:** {session.get('date', 'Unknown')}")
                st.write(f"**Duration:** {session.get('duration', 'Unknown')}")
                st.write(f"**Interactions:** {session.get('interaction_count', 0)}")
                st.write(f"**Status:** {'Completed' if session.get('completed', False) else 'Incomplete'}")
                
                if session.get('summary'):
                    st.write("**Session Summary:**")
                    st.write(session['summary'])
            
            with col2:
                if st.button(f"Restart Topic", key=f"restart_{i}"):
                    st.session_state.selected_topic_for_revision = session.get('topic')
                    st.switch_page("app.py")
                
                if st.button(f"Delete Session", key=f"delete_{i}"):
                    st.session_state.session_history.pop(-(i+1))
                    st.rerun()