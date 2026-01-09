"""GLM-4.7 LangChain compatible client for Zhipu AI."""
from typing import Any, List, Optional, Iterator
import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    AIMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field, SecretStr
from tenacity import retry, stop_after_attempt, wait_exponential


class GLMChat(BaseChatModel):
    """LangChain compatible chat model for Zhipu AI GLM-4.7."""

    api_key: SecretStr = Field(..., description="Zhipu AI API key")
    base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4/",
        description="API base URL"
    )
    model: str = Field(default="glm-4.7", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=4096)
    timeout: int = Field(default=60)

    @property
    def _llm_type(self) -> str:
        return "glm-4.7"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to API format."""
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted.append({"role": "assistant", "content": msg.content})
            else:
                converted.append({"role": "user", "content": str(msg.content)})
        return converted

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_api(self, messages: List[dict]) -> dict:
        """Make API call with retry logic."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion."""
        converted_messages = self._convert_messages(messages)
        response = self._call_api(converted_messages)

        content = response["choices"][0]["message"]["content"]
        message = AIMessage(content=content)

        generation = ChatGeneration(
            message=message,
            generation_info={
                "finish_reason": response["choices"][0].get("finish_reason"),
                "usage": response.get("usage", {}),
            },
        )

        return ChatResult(
            generations=[generation],
            llm_output={"model": self.model, "usage": response.get("usage", {})},
        )


class GLMEmbeddings(Embeddings):
    """LangChain compatible embeddings for Zhipu AI."""

    api_key: SecretStr = Field(..., description="Zhipu AI API key")
    base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4/",
        description="API base URL"
    )
    model: str = Field(default="embedding-3", description="Embedding model name")
    timeout: int = Field(default=60)

    def __init__(self, **kwargs):
        # Pydantic v2 style initialization
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Make embedding API call."""
        url = f"{self.base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        # Sort by index to ensure correct order
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        # Process in batches to avoid API limits
        batch_size = 16
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._call_api([text])[0]
