import boto3
import dotenv
import json
from datetime import datetime
from chatmessage import ChatMessage
from sources import sources

dotenv.load_dotenv()

json_value = {
    "query_area": "",
    "enterprise_name": "",
    "enterprise_group": "",
    "farm_id": "F7A3B22B-0368-2B28-E995-917FBE2A8BE1",
    "user_id": "",
    "application": "",
    "question": "How many animals do I have?",
    "data": ""
}
agent_id = "CVH9H4JPAX"
agent_alias_id = "T0S6C9LJ7I"
prompt = json.dumps(json_value)
session_id = "session1"

agent = boto3.client(service_name='bedrock-agent-runtime', region_name='eu-west-1')

#modelarm = 'arn:aws:bedrock:eu-west-1:513382409702:inference-profile/eu.anthropic.claude-3-5-sonnet-20240620-v1:0'
modelarm = 'arn:aws:bedrock:eu-west-1:513382409702:inference-profile/eu.anthropic.claude-3-5-sonnet-20240620-v1:0'
knowledgeBaseId = 'BKWDTDREEZ'

maxTokens = 2000
temperature = 0.7
topP = 0.9
MAX_MESSAGES = 30

def filter_citations(citations):
    new_cit = []
    for citation in citations:
        text = citation['generatedResponsePart']['textResponsePart']['text']
        start = citation['generatedResponsePart']['textResponsePart']['span']['start']
        end = citation['generatedResponsePart']['textResponsePart']['span']['end']
        links = []

        if not citation['retrievedReferences']:  # Safer check
            continue

        for link in citation['retrievedReferences']:
            if link['location']['type'] == 'S3':
                links.append({
                    "type": "S3",
                    "text": link['location']['s3Location']['uri'].split('/')[-1],
                    "url": sources.get(link['location']['s3Location']['uri'].split('/')[-1], "")
                })
            else:
                links.append({
                    "type": "WEB",
                    "text": link['content']['text'].split('|')[0],
                    "url": link['location'].get('webLocation', {}).get('url', '').replace(':443', '').replace('http://', 'https://')
                })

        new_cit.append({
            "text": text,
            "start": start,
            "end": end,
            "links": list({d['url']: d for d in links}.values())  # Remove duplicates
        })
    return new_cit

def add_citations_to_text(chatMessage):
    new_text = chatMessage.text
    ordered_citations = sorted(chatMessage.citations, key=lambda x: x["start"], reverse=True)
    for citation in ordered_citations:
        new_text = new_text[:min(citation['end']+1, len(new_text))] + f' [Read More...]( {citation["links"][0]["url"]}) \n\n' + new_text[min(citation['end']+1, len(new_text)):]
    return ChatMessage('assistant', new_text)

def run_query_with_knowledge_base(promt, session_id=None):
    if session_id is None:
        response = agent.retrieve_and_generate(
            input={'text': promt},
            retrieveAndGenerateConfiguration={
                'knowledgeBaseConfiguration': {
                    'generationConfiguration': {
                        'inferenceConfig': {
                            'textInferenceConfig': {
                                'maxTokens': maxTokens,
                                'temperature': temperature,
                                'topP': topP
                            }
                        }
                    },
                    'knowledgeBaseId': knowledgeBaseId,
                    'modelArn': modelarm,
                    'orchestrationConfiguration': {
                        'queryTransformationConfiguration': {
                            'type': 'QUERY_DECOMPOSITION'
                        }
                    }
                },
                'type': 'KNOWLEDGE_BASE'
            }
        )
    else:
        bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
        response = bedrock_agent_runtime_client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=promt,
            streamingConfigurations={"enableCompletionStreaming": True}  # Enable Streaming
        )

    print(response)  # Debugging

    # Ensure `output` is safely assigned before use
    output = response.get('output', {}).get('text', "No response text available")

    # Remove or transform `EventStream` before JSON serialization
    if "completion" in response:
        response["completion"] = str(response["completion"])  # Convert EventStream to string

    try:
        response_json = json.dumps(response, indent=4)
        print(response_json)  # Debugging output
    except TypeError as e:
        print(f"Error serializing response: {e}")

    # Ensure 'citations' key exists before accessing
    citations = response.get('citations', [])

    return (add_citations_to_text(ChatMessage('assistant', output, filter_citations(citations))), response.get('sessionId', "UnknownSession"))

def chat_with_model(message_history, new_text, session_id=None):
    """Main output function."""
    new_text_message = ChatMessage('user', text=new_text)
    message_history.append(new_text_message)

    number_of_messages = len(message_history)

    if number_of_messages > MAX_MESSAGES:
        del message_history[0 : (number_of_messages - MAX_MESSAGES) * 2] 

    response_message, session_id = run_query_with_knowledge_base(new_text, session_id)

    message_history.append(response_message)

    return session_id
