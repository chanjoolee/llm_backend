from abc import ABC
from enum import Enum

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class ComponentType(Enum):
    PROMPT = "pt"
    TOOL = "tl"
    DATASOURCE = "ds"
    AGENT = "ag"


class Component(BaseModel, ABC):
    """
    모든 컴포넌트의 부모 클래스입니다.
    """
    name: str
    description: str


def create_tool_name(component_type: ComponentType, name: str, username: str) -> str:
    return f"{name}_{component_type.value}_{username}"

