import json

class DocumentEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, object) and hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

class Document:
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata
