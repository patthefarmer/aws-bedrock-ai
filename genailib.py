import boto3
import json
import sys
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv
from chatmessage import ChatMessage
from sources import sources

# Load environment variables
load_dotenv()

# ⚡ Direct Bedrock Model Configuration
model_id = "eu.anthropic.claude-3-5-sonnet-20240620-v1:0"

# ⚡ Optimized AI Model Parameters
max_tokens_to_sample = 2000
temperature = 0.6
top_p = 0.5
MAX_MESSAGES = 20  # ✅ Keeps chat history manageable

# ⚡ Initialize Bedrock Client
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="eu-west-1")

# 🚀 Streaming AI Model Query with Memory (FIXED DUPLICATION)
def run_query_with_ai_model(prompt, message_history, session_id=None):
    """Queries Amazon Bedrock AI Model using a streaming response with memory."""
    print("\n🔍 **Calling Bedrock AI Model - Streaming Response...**\n")
    session_id = session_id or "default-session"

    # ✅ Keep last `MAX_MESSAGES * 2` exchanges for memory, **excluding empty messages**
    messages = [
        {"role": msg.role, "content": msg.text}
        for msg in message_history[-MAX_MESSAGES * 2:] if msg.text.strip()
    ]

    # ✅ Append current user message
    messages.append({"role": "user", "content": prompt})

    try:
        # ✅ Use `invoke_model_with_response_stream()` for real-time streaming
        response = bedrock_client.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "system": "You are a livestock advisory for the Herdwatch livestock management app.\nHere are some relevant sources to check first:\n'https://help.herdwatch.com/en/'\n'https://herdwatch.com/'\n",
                "messages": messages,  # ✅ Pass full conversation history, **excluding empty messages**
                "max_tokens": max_tokens_to_sample,
                "temperature": temperature,
                "top_p": top_p
            })
        )

        # ✅ Initialize streaming output
        print("\n🟢 **Streaming response from Claude...**\n")
        streamed_response = ""

        for event in response["body"]:
            print(f"🔹 Raw event received: {event}")

            if "chunk" in event and "bytes" in event["chunk"]:
                chunk_data = event["chunk"]["bytes"].decode("utf-8")  # Decode bytes
                print(f"🔹 Decoded chunk: {chunk_data}")

                try:
                    chunk_json = json.loads(chunk_data)
                    if "content_block_delta" in chunk_json.get("type", ""):
                        text_chunk = chunk_json["delta"].get("text", "")

                        streamed_response += text_chunk
                        yield text_chunk  # ✅ Stream **each chunk dynamically** (NO DUPLICATION)

                except json.JSONDecodeError as json_err:
                    print(f"⚠️ JSON Decode Error: {json_err}")

    except Exception as e:
        print(f"❌ Error calling Bedrock AI Model: {str(e)}")
        yield "An error occurred while processing your request."

# 🚀 Optimized Chat Function with Memory (NO DUPLICATION)
def chat_with_model(message_history, new_text, session_id=None):
    """Handles AI chat session with memory and streaming."""

    # ✅ Append user message **once**
    new_text_message = ChatMessage("user", new_text)
    message_history.append(new_text_message)

    # ✅ Ensure history doesn't exceed limits
    if len(message_history) > MAX_MESSAGES * 2:
        del message_history[: len(message_history) - MAX_MESSAGES * 2]

    # ✅ **Fixed**: Don't add an empty assistant message
    assistant_message = ChatMessage("assistant", "...")
    message_history.append(assistant_message)  # ✅ Use placeholder text instead of empty string

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_model = executor.submit(run_query_with_ai_model, new_text, message_history, session_id)

        # ✅ Stream the response live (NO DUPLICATION)
        streamed_response = ""
        for response_chunk in future_model.result():
            streamed_response += response_chunk
            yield response_chunk  # ✅ Stream dynamically, **NO extra response at the end**

        # ✅ Instead of appending another message, update the assistant message
        message_history[-1].text = streamed_response  # ✅ Modify last assistant message instead of adding a new one

        # ✅ Debugging - Check final stored message
        print(f"🔍 Final Assistant Message: {message_history[-1].text}")
