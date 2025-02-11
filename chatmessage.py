import json

class ChatMessage:
    def __init__(self, role, text, citations=None):
        self.role = role
        self.text = text
        self.citations = citations

    def to_dict(self):
        return {
            'role': self.role,
            'text': self.text,
            'citations': self.citations
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            role=data['role'],
            text=data['text'],
            citations=data.get('citations')
        )

# Функції для серіалізації масиву об'єктів у JSON і навпаки
def serialize_messages(messages):
    """Перетворює масив об'єктів ChatMessage на JSON."""
    return json.dumps([message.to_dict() for message in messages])

def deserialize_messages(json_str):
    """Перетворює JSON-рядок на масив об'єктів ChatMessage."""
    if json_str is None or json_str == "":
        return []
    data = json.loads(json_str)
    return [ChatMessage.from_dict(item) for item in data]

# Приклад використання
messages = [
    ChatMessage(role="user", text="Hello!", citations=["Ref1"]),
    ChatMessage(role="assistant", text="Hi there!", citations=None)
]
