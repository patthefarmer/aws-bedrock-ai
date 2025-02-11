import streamlit as st
from streamlit_local_storage import LocalStorage

import os
import random
import json

import genailib
import chatmessage

localS = LocalStorage()

# Изменение фона на светло-зеленый
st.markdown(
    """
    <style>
    html, body, .stApp {
        background-color: #d0f0c0;  /* Светло-зеленый цвет */
        height: 100%;
        margin: 0;
        padding: 0;
    }
    h1, h2, .stSubheader {
        color: #000000;  /* Черный цвет шрифта для заголовков и подзаголовков */
    }
    .stMarkdown, .stChatMessage {
        color: #000000;  /* Черный цвет шрифта для текста сообщений */
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
#col1.image("logo-footer-1.svg", caption="", width=300) 
clearbutton = col2.button("Clear History")

if clearbutton:
    localS.deleteAll()
    localS.setItem("sessionId", None, key='set-session')
    st.session_state.messages = []
    localS.setItem("chat-history","[]", key='set-chat')

st.title('Herdbot')

# Черный цвет для текста, но радуга для разделителя
st.markdown("<h2 style='color: black;'>Your Farm, Smarter with AI</h2>", unsafe_allow_html=True)
st.markdown('<div class="rainbow-divider"></div>', unsafe_allow_html=True)

st.session_state.messages = chatmessage.deserialize_messages(localS.getItem("chat-history"))
if st.session_state.messages is None:
    st.session_state.messages = []

@st.cache_data
def get_welcome_message() -> str:
    return random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi there! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )

def get_history() -> str:
    hisotry_list = [
        f"{record['role']}: {record['content']}" for record in st.session_state.messages
    ]
    return '\n\n'.join(hisotry_list)

if "messages" not in st.session_state:
    st.session_state.messages = []

input_text = st.chat_input("Chat with your bot here") 

if input_text:
    session_id = genailib.chat_with_model(message_history=st.session_state.messages, new_text=input_text, session_id=localS.getItem("sessionId"))
    localS.deleteAll()
    localS.setItem("sessionId", session_id, key='set-session')
    localS.setItem("chat-history", chatmessage.serialize_messages(st.session_state.messages), key='set-chat')
welcome_message = get_welcome_message()
with st.chat_message('assistant'):
    st.markdown(welcome_message)
for message in st.session_state.messages:
    with st.chat_message(message.role):
        st.markdown(message.text)
        if message.citations:
            for cit in message.citations:
                for link in cit['links']:
                    st.write(f"[{link['text']}]({link['url']})")

if DEBUG := os.getenv("DEBUG", False):
    st.subheader("History", divider="rainbow")
    history_list = [
        f"{record.role}: {record.text}, {record.citations}" for record in st.session_state.messages
    ]
    st.write(history_list)