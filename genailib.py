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

# ⚡ Bedrock Model Configuration
model_id = "eu.anthropic.claude-3-5-sonnet-20240620-v1:0"
knowledge_base_id = "BKWDTDREEZ"  # ✅ AWS Knowledge Base ID

# ⚡ Optimized AI Model Parameters
max_tokens_to_sample = 2000
temperature = 0.4
top_p = 0.6
MAX_MESSAGES = 20  # ✅ Keeps chat history manageable

# ⚡ Initialize AWS Clients
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="eu-west-1")
kb_client = boto3.client(service_name="bedrock-agent-runtime", region_name="eu-west-1")

# 🚀 Streaming Query to AWS Knowledge Base (Primary Source)
def query_knowledge_base_stream(prompt, message_history, session_id=None):
    """Queries AWS Knowledge Base using streaming response before calling Claude."""
    print("\n🔍 **Checking AWS Knowledge Base First...**\n")
    session_id = session_id or "default-session"

    # ✅ Include previous questions & answers
    history_text = "\n".join([f"{msg.role}: {msg.text}" for msg in message_history])

    full_query = f"Previous conversation:\n{history_text}\nNew question:\n{prompt}"

    try:
        response = kb_client.retrieve_and_generate_stream(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery=full_query  # ✅ Use full chat history
        )

        print("\n🟢 **Streaming response from AWS Knowledge Base...**\n")
        streamed_response = ""

        for event in response["body"]:
            if "chunk" in event and "bytes" in event["chunk"]:
                chunk_data = event["chunk"]["bytes"].decode("utf-8")

                try:
                    chunk_json = json.loads(chunk_data)
                    if "outputText" in chunk_json:
                        text_chunk = chunk_json["outputText"]
                        streamed_response += text_chunk
                        yield text_chunk  # ✅ Stream from KB

                except json.JSONDecodeError as json_err:
                    print(f"⚠️ JSON Decode Error from KB: {json_err}")

        if streamed_response.strip():
            return  # ✅ Stop here if KB returns a valid response

    except Exception as e:
        print(f"⚠️ AWS Knowledge Base query failed: {str(e)}")

    # 🚀 If KB has no useful response, call Claude **with chat history**
    print("\n⚠️ No useful response from KB, switching to Claude AI Model...\n")
    yield from run_query_with_ai_model(prompt, message_history, session_id)

# 🚀 Streaming AI Model Query (Fallback to Claude)
def run_query_with_ai_model(prompt, message_history, session_id=None):
    """Queries Amazon Bedrock AI Model using a streaming response with memory."""
    print("\n🔍 **Calling Bedrock AI Model - Streaming Response...**\n")
    session_id = session_id or "default-session"

    # ✅ Convert chat history to Claude's message format
    messages = [
        {"role": msg.role, "content": msg.text}
        for msg in message_history[-MAX_MESSAGES * 2:] if msg.text.strip()
    ]
    messages.append({"role": "user", "content": prompt})  # ✅ Include current user input

    try:
        response = bedrock_client.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "system": "You are a livestock advisory for the Herdwatch livestock management app.\nHere are some relevant sources to check first:\n'https://help.herdwatch.com/en/'\n'https://herdwatch.com/'\n",
                "messages": messages,  # ✅ Pass full conversation history
                "max_tokens": max_tokens_to_sample,
                "temperature": temperature,
                "top_p": top_p
            })
        )

        print("\n🟢 **Streaming response from Claude...**\n")
        streamed_response = ""

        for event in response["body"]:
            if "chunk" in event and "bytes" in event["chunk"]:
                chunk_data = event["chunk"]["bytes"].decode("utf-8")

                try:
                    chunk_json = json.loads(chunk_data)
                    if "content_block_delta" in chunk_json.get("type", ""):
                        text_chunk = chunk_json["delta"].get("text", "")
                        streamed_response += text_chunk
                        yield text_chunk  # ✅ Stream dynamically

                except json.JSONDecodeError as json_err:
                    print(f"⚠️ JSON Decode Error from Claude: {json_err}")

    except Exception as e:
        print(f"❌ Error calling Bedrock AI Model: {str(e)}")
        yield "An error occurred while processing your request."

# 🚀 Optimized Chat Function with Knowledge Base + Claude AI
def chat_with_model(message_history, new_text, session_id=None):
    """Handles AI chat session with memory, streaming, and AWS Knowledge Base check."""

    new_text_message = ChatMessage("user", new_text)
    message_history.append(new_text_message)

    # ✅ Trim history to avoid excessive memory usage
    if len(message_history) > MAX_MESSAGES * 2:
        del message_history[: len(message_history) - MAX_MESSAGES * 2]

    # ✅ Add placeholder for assistant's response
    assistant_message = ChatMessage("assistant", "...")
    message_history.append(assistant_message)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_model = executor.submit(query_knowledge_base_stream, new_text, message_history, session_id)

        streamed_response = ""
        for response_chunk in future_model.result():
            streamed_response += response_chunk
            yield response_chunk  # ✅ Stream dynamically from KB or Claude

        # ✅ Store full assistant response
        message_history[-1].text = streamed_response
        print(f"🔍 Final Assistant Message: {message_history[-1].text}")
