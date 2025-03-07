from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ai_core.llm_api_provider import LlmApiProvider


def _create_openai_chat_model(llm_api_key: str, llm_api_url: str, llm_model: str,
                              temperature: float = 0.2, max_tokens: int = 1024):
    return ChatOpenAI(
        model=llm_model,
        openai_api_key=llm_api_key,
        openai_api_base=llm_api_url,
        temperature=temperature,
        max_tokens=max_tokens
    )


def _create_anthropic_chat_model(llm_api_key: str, llm_api_url: str, llm_model: str,
                                 temperature: float = 0.2, max_tokens: int = 1024):
    return ChatAnthropic(
        model=llm_model,
        base_url=llm_api_url,
        api_key=llm_api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )


def create_chat_model(llm_api_provider: str = "ai_one",
                      llm_api_key: str = "...",
                      llm_api_url: str = "https://api.platform.a15t.com/v1",
                      llm_model: str = "anthropic/claude-3-5-sonnet-20240620",
                      temperature: float = 0.2,
                      max_tokens: int = 1024) -> BaseChatModel:
    if llm_api_provider == LlmApiProvider.SMART_BEE.value:
        return _create_openai_chat_model(llm_api_key, llm_api_url, llm_model, temperature, max_tokens)
    elif llm_api_provider == LlmApiProvider.AI_ONE.value:
        return _create_anthropic_chat_model(llm_api_key, llm_api_url, llm_model, temperature, max_tokens)

    elif llm_api_provider == LlmApiProvider.OPENAI.value:
        return _create_openai_chat_model(llm_api_key, llm_api_url, llm_model, temperature, max_tokens)
        # if llm_model.startswith('openai/'):
        #     return _create_openai_chat_model(llm_api_key, llm_api_url, llm_model, temperature, max_tokens)
        # elif llm_model.startswith('anthropic/'):
        #     return _create_anthropic_chat_model(llm_api_key, llm_api_url, llm_model, temperature, max_tokens)

    raise ValueError("Invalid LLM provider")

