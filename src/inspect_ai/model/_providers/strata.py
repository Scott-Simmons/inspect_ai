import os
from typing import Any

from .._generate_config import GenerateConfig
from .openai_compatible import OpenAICompatibleAPI


class StrataAPI(OpenAICompatibleAPI):
    """Connect to an existing Strata OpenAI-compatible server."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key or os.environ.get("STRATA_API_KEY") or "strata",
            config=config,
            service="strata",
            service_base_url="http://localhost:8080/v1",
            **model_args,
        )
