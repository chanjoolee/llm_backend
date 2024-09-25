import os
from importlib.machinery import SourceFileLoader

from langchain_core.tools import StructuredTool

from ai_core.base import ComponentType, create_tool_name


def load_tool(name, username, tool_path):
    filename = os.path.basename(tool_path)
    module_name = os.path.splitext(filename)[0]
    loader = SourceFileLoader(module_name, tool_path)
    module = loader.load_module()

    attributes = dir(module)

    tools = [getattr(module, attr) for attr in attributes if isinstance(getattr(module, attr), StructuredTool)]
    if len(tools) != 1:
        raise ValueError(f"Invalid number of tools found in {tool_path}")

    # 사용자와 컴포넌트 타입 사이에 이름 중복을 피하기 위해 새로운 이름을 부여
    tools[0].name = create_tool_name(ComponentType.TOOL, name, username)

    return tools[0]
