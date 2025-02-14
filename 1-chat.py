import streamlit as st
from streamlit_local_storage import LocalStorage
import os
import random
import json
import genailib
import chatmessage

localS = LocalStorage()

st.markdown(
    """
    <style>
    html, body, .stApp {
        background-color: #d0f0c0;
        height: 100%;
        margin: 0;
        padding: 0;
    }
    h1, h2, .stSubheader {
        color: #000000;
    }
    .stMarkdown, .stChatMessage {
        color: #000000;
    }
    .rainbow-divider {
        background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
        height: 2px;
        border: none;
    }
    .st-key-set, .st-key-deleteAll{
        height: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")
clearbutton = col2.button("Clear History", key="clear-history-btn")

if clearbutton:
    localS.deleteAll()
    localS.setItem("sessionId", None, key='set-session')
    st.session_state.messages = []
    localS.setItem("chat-history", "[]", key="set-chat-history")

st.title('Herdbot')
st.markdown("<h2 style='color: black;'>Your Farm, Smarter with AI</h2>", unsafe_allow_html=True)
st.markdown('<div class="rainbow-divider"></div>', unsafe_allow_html=True)

# ✅ Load previous messages (Ensure they are in order)
if "messages" not in st.session_state:
    st.session_state.messages = chatmessage.deserialize_messages(localS.getItem("chat-history")) or []

@st.cache_data
def get_welcome_message() -> str:
    return random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi there! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )

# ✅ Display existing chat messages **before** new input
for message in st.session_state.messages:
    with st.chat_message(message.role):
        st.markdown(message.text)
        if hasattr(message, "citations") and message.citations:
            for cit in message.citations:
                for link in cit.get("links", []):
                    st.write(f"[{link['text']}]({link['url']})")

# ✅ User Input Box
input_text = st.chat_input("Chat with your bot here")

if input_text:
    # ✅ Display user's question immediately **before streaming starts**
    with st.chat_message("user"):
        st.markdown(input_text)

    # ✅ Append user message to chat history before streaming
    user_message = chatmessage.ChatMessage("user", input_text)
    st.session_state.messages.append(user_message)

    # ✅ Placeholder assistant message (prevents ordering issues)
    assistant_message = chatmessage.ChatMessage("assistant", "")
    st.session_state.messages.append(assistant_message)

    # ✅ Save session ID **before** streaming starts
    if "sessionId" not in st.session_state:
        st.session_state.sessionId = localS.getItem("sessionId")

    # ✅ Stream response into placeholder message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        streamed_response = ""

        for response_chunk in genailib.chat_with_model(
            message_history=st.session_state.messages[:-1],  # ✅ Send history **excluding** placeholder
            new_text=input_text,
            session_id=st.session_state.sessionId
        ):
            streamed_response += response_chunk
            message_placeholder.markdown(streamed_response)

        # ✅ Update the **last assistant message** in session state
        st.session_state.messages[-1].text = streamed_response  

    # ✅ Save chat history after streaming (Fixed key, single call)
    if "sessionId" in st.session_state:
        localS.setItem("sessionId", st.session_state.sessionId, key="set-session")
    localS.setItem("chat-history", chatmessage.serialize_messages(st.session_state.messages), key="set-chat-history")

    # ✅ Force Streamlit to **re-render messages in correct order**
    st.rerun()

# ✅ Display Welcome Message (Only if no messages exist)
if len(st.session_state.messages) == 0:
    welcome_message = get_welcome_message()
    with st.chat_message('assistant'):
        st.markdown(welcome_message)

# ✅ Debug Mode (Optional)
if DEBUG := os.getenv("DEBUG", False):
    st.subheader("History", divider="rainbow")
    history_list = [
        f"{record.role}: {record.text}, {record.citations}" for record in st.session_state.messages
    ]
    st.write(history_list)
