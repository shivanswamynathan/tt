import streamlit as st
import requests
import json
import websocket
import threading
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
    st.session_state.session_complete = False
if "cached_topics" not in st.session_state:
    st.session_state.cached_topics = None
if "topics_loaded" not in st.session_state:
    st.session_state.topics_loaded = False
# WebSocket state variables
if "ws_connected" not in st.session_state:
    st.session_state.ws_connected = False
if "ws_client" not in st.session_state:
    st.session_state.ws_client = None

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
                
                # WebSocket connection status
                if st.session_state.ws_connected:
                    st.success("🔗 WebSocket Connected")
                else:
                    st.error("❌ WebSocket Disconnected")
                
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
    """Show the main revision chat interface"""
    st.header(f"📖 Revising: {st.session_state.current_topic}")
    
    # Show current stage and conversation count
    if st.session_state.conversation_count > 0:
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
    """Fetch available topics from the API - with caching"""
    try:
        # Only fetch if not cached
        if st.session_state.get("cached_topics") is not None:
            return st.session_state.cached_topics
            
        response = requests.get(f"{API_BASE_URL}/topics")
        if response.status_code == 200:
            topics = response.json()["topics"]
            st.session_state.cached_topics = topics
            return topics
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
        st.session_state.ws_connected = False
        st.session_state.ws_client = None
        
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
            
            # Connect WebSocket after initial setup
            connect_websocket()
            
            st.success(f"✅ Started revision session for '{topic}'!")
            st.rerun()
        else:
            st.error(f"Failed to start session: {response.json().get('detail', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"Error starting session: {str(e)}")

def connect_websocket():
    """Connect to WebSocket for chat"""
    try:
        ws_url = f"ws://localhost:8000/api/ws/revision/{st.session_state.session_id}"
        print(f"Connecting to: {ws_url}")
        
        def on_message(ws, message):
            """Handle incoming WebSocket messages"""
            try:
                print(f"Received message: {message}")
                data = json.loads(message)
                
                if data["type"] == "message":
                    # Store message in a simple way
                    if "pending_messages" not in st.session_state:
                        st.session_state.pending_messages = []
                    
                    assistant_message = {
                        "role": "assistant",
                        "content": data["content"],
                        "metadata": {
                            "conversation_count": data.get("conversation_count", 0),
                            "sources": data.get("sources", []),
                            "is_session_complete": data.get("is_session_complete", False),
                            "current_stage": data.get("current_stage")
                        }
                    }
                    st.session_state.pending_messages.append(assistant_message)
                    print(f"Message added to pending queue")
                    
            except Exception as e:
                print(f"Error processing message: {e}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        def on_open(ws):
            print(f"WebSocket connected successfully!")
            # Don't try to set session state here
        
        # Create WebSocket connection
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Store WebSocket client
        st.session_state.ws_client = ws
        
        # Run WebSocket in background thread
        def run_ws():
            ws.run_forever()
        
        ws_thread = threading.Thread(target=run_ws)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Mark as connected
        st.session_state.ws_connected = True
        
    except Exception as e:
        print(f"Failed to connect WebSocket: {e}")
        st.error(f"Failed to connect WebSocket: {e}")

def handle_user_input(user_input):
    """Send message via WebSocket"""
    try:
        if not st.session_state.ws_client:
            st.error("WebSocket not connected. Please refresh the page.")
            return
        
        # Add user message to chat immediately
        user_message = {"role": "user", "content": user_input}
        st.session_state.revision_messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Send message via WebSocket
        st.session_state.ws_client.send(user_input)
        print(f"Sent message: {user_input}")
        
        # Check for pending messages and add them
        if "pending_messages" in st.session_state and st.session_state.pending_messages:
            for msg in st.session_state.pending_messages:
                st.session_state.revision_messages.append(msg)
                st.session_state.conversation_count = msg["metadata"].get("conversation_count", st.session_state.conversation_count)
            
            # Clear pending messages
            st.session_state.pending_messages = []
            st.rerun()
        
    except Exception as e:
        print(f"Error sending message: {e}")
        st.error(f"Error sending message: {str(e)}")

def end_session():
    """End the current revision session"""
    # Close WebSocket connection
    if st.session_state.ws_client:
        st.session_state.ws_client.close()
    
    # Reset session state
    st.session_state.session_id = None
    st.session_state.current_topic = None
    st.session_state.conversation_count = 0
    st.session_state.revision_messages = []
    st.session_state.session_complete = False
    st.session_state.ws_connected = False
    st.session_state.ws_client = None
    
    st.success("Session ended. You can start a new revision session anytime!")
    st.rerun()

if __name__ == "__main__":
    main()