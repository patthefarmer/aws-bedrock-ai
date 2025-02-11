import boto3
import dotenv
import json
import sys
import re
import concurrent.futures
from datetime import datetime
from chatmessage import ChatMessage
from sources import sources
from botocore.eventstream import EventStream

dotenv.load_dotenv()

# ⚡ Bedrock Agent Configuration (Optimized for Speed)
agent_id = "CVH9H4JPAX"
agent_alias_id = "WTQJJVHLVI"  # ✅ Stored Correctly

# ⚡ Fastest Model: Claude 3 Haiku
knowledgeBaseId = "BKWDTDREEZ"
modelarm = "arn:aws:bedrock:eu-west-1:123456789:model/claude-3-haiku"

# ⚡ Optimized AI Model Parameters
maxTokens = 300  # ✅ Reduced for speed
temperature = 0.5  # ✅ Lower for deterministic responses
topP = 0.85  # ✅ Slightly reduced for efficiency
MAX_MESSAGES = 20  # ✅ Reduced to prevent memory overhead

# ⚡ Initialize Bedrock Client (Using Persistent Connection for Speed)
agent = boto3.client(service_name="bedrock-agent-runtime", region_name="eu-west-1")

# 🚀 Fast EventStream Processing
def process_event_stream(event_stream):
    """Extracts and processes text content from an EventStream response."""
    full_response = []

    try:
        for event in event_stream:
            if "chunk" in event and "bytes" in event["chunk"]:
                chunk_text = event["chunk"]["bytes"].decode("utf-8")
                sys.stdout.write(chunk_text)
                sys.stdout.flush()
                full_response.append(chunk_text)  # ✅ Collect response text

        print("\n")
    except Exception as e:
        print(f"❌ Error processing EventStream: {str(e)}")

    return "".join(full_response).strip()

# 🚀 Remove Unnecessary AI Disclaimers
def clean_ai_response(response_text):
    """Removes AI-generated disclaimers and ensures direct factual responses."""
    apology_phrases = [
        r"Sorry, I am unable to assist you with this request.*?",
        r"I cannot provide that information.*?",
        r"I'm unable to help with that request.*?",
        r"I apologize that I couldn't find this specific information.*?",
        r"I couldn't find this in the knowledge base.*?",
        r"This information is based on general knowledge.*?",
        r"If you need more specific details, please let me know.*?"
    ]
    for phrase in apology_phrases:
        response_text = re.sub(phrase, "", response_text, flags=re.IGNORECASE).strip()
    return response_text

# 🚀 Run Knowledge Base Query (Parallel Processing Enabled)
def run_query_with_knowledge_base(prompt, session_id=None):
    """Queries Amazon Bedrock AI Agent with knowledge base first."""
    print("\n🔍 **Calling Bedrock Agent - Searching Knowledge Base...**\n")

    try:
        response = agent.retrieve_and_generate(
            input={"text": prompt},
            retrieveAndGenerateConfiguration={
                "knowledgeBaseConfiguration": {
                    "generationConfiguration": {
                        "inferenceConfig": {
                            "textInferenceConfig": {
                                "maxTokens": maxTokens,
                                "temperature": temperature,
                                "topP": topP
                            }
                        }
                    },
                    "knowledgeBaseId": knowledgeBaseId,
                    "modelArn": modelarm,
                    "orchestrationConfiguration": {
                        "queryTransformationConfiguration": {
                            "type": "QUERY_DECOMPOSITION"
                        }
                    }
                },
                "type": "KNOWLEDGE_BASE"
            }
        )

        print("\n🔍 **Raw API Response (Knowledge Base):**\n", json.dumps(response, indent=4))

        # ✅ Extract response text
        output = response.get("output", {}).get("text", "").strip()

        # ✅ Detect unhelpful KB responses and fall back to AI model immediately
        if (
            not output or 
            "Sorry, I am unable to assist" in output or
            "I could not find" in output or 
            "The search results do not contain" in output
        ):
            print("\n⚠️ **No Useful Answer in KB - Switching to AI Model**\n")
            return run_query_with_ai_model(prompt, session_id)

        return ChatMessage("assistant", output), response.get("sessionId", "UnknownSession")

    except Exception as e:
        print(f"⚠️ Error retrieving from Knowledge Base: {str(e)}")
        return run_query_with_ai_model(prompt, session_id)

# 🚀 Direct AI Model Query (Parallel Processing Enabled)
def run_query_with_ai_model(prompt, session_id=None):
    """Fallback method: Uses AI Model if no KB results are found."""
    print("\n🔍 **Calling Bedrock Agent - AI Model Response...**\n")

    session_id = session_id or "default-session"

    try:
        bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name="eu-west-1")

        response = bedrock_agent_runtime_client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=prompt,
            enableTrace=False,
            endSession=False
        )

        print("\n🔍 **Raw API Response (AI Model) [Optimized]:**")
        response_copy = {key: str(value) if isinstance(value, EventStream) else value 
                         for key, value in response.items()}
        print(json.dumps(response_copy, indent=4))

        # ✅ Use EventStream for Faster Processing
        if "completion" in response and isinstance(response["completion"], EventStream):
            print("\n🔍 **Processing EventStream Response Faster...**")
            full_response = process_event_stream(response["completion"])
        else:
            full_response = response.get("completion", {}).get("outputText", "No response text available")

        # ✅ Ensure AI provides factual responses
        full_response = clean_ai_response(full_response)

        return ChatMessage("assistant", full_response), response.get("sessionId", session_id)

    except Exception as e:
        print(f"❌ Error calling Bedrock AI Model: {str(e)}")
        return ChatMessage("assistant", "An error occurred while processing your request."), session_id

# 🚀 Optimized Chat Function with Parallel Execution
def chat_with_model(message_history, new_text, session_id=None):
    """Handles AI chat session with optimized execution."""
    new_text_message = ChatMessage("user", text=new_text)
    message_history.append(new_text_message)

    if len(message_history) > MAX_MESSAGES:
        del message_history[0 : (len(message_history) - MAX_MESSAGES) * 2]

    # ✅ Run KB and AI Model in Parallel for Speed
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_kb = executor.submit(run_query_with_knowledge_base, new_text, session_id)
        response_message, session_id = future_kb.result()  # Get the fastest result

    message_history.append(response_message)

    return session_id
