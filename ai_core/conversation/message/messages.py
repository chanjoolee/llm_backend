from dataclasses import dataclass

from ai_core.conversation.message.base import DaisyMessage, DaisyMessageRole


@dataclass
class DaisySystemMessage(DaisyMessage):
    role = DaisyMessageRole.SYSTEM


@dataclass
class DaisyAIMessage(DaisyMessage):
    role = DaisyMessageRole.AI


@dataclass
class DaisyUserMessage(DaisyMessage):
    role = DaisyMessageRole.HUMAN
