from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr
import requests
from typing import Any
from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs import BoolInput, DictInput, DropdownInput, IntInput, SecretStrInput, SliderInput, StrInput
from langflow.schema.dotdict import dotdict

#gpustack  plugin  in langflow
class GPUStackModelComponent(LCModelComponent):
    display_name = "GPUStack"
    description = "Generates text using GPUStack LLMs."
    icon = "GPUStack"
    name = "GPUStackModel"

    inputs = [
        *LCModelComponent._base_inputs,
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
            range_spec=RangeSpec(min=0, max=128000),
        ),
        DictInput(
            name="model_kwargs",
            display_name="Model Kwargs",
            advanced=True,
            info="Additional keyword arguments to pass to the model.",
        ),
        BoolInput(
            name="json_mode",
            display_name="JSON Mode",
            advanced=True,
            info="If True, it will output JSON regardless of passing a schema.",
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            advanced=False,
            options=[],
            required=True,
            combobox=True,
            refresh_button=True,
        ),
        StrInput(
            name="gpustack_api_base",
            display_name="GPUStack API Base",
            advanced=False,
            info="The base URL of the GPUStack API. "
            "Defaults to https://api.gpustack.com/v1. "
            "You can change this to use other APIs like JinaChat, LocalAI and Prem.",
            refresh_button=True,
            required=True,
            real_time_refresh=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="GPUStack API Key",
            info="The GPUStack API Key to use for the GPUStack model.",
            advanced=False,
            value="OPENAI_API_KEY",
        ),
        SliderInput(
            name="temperature", display_name="Temperature", value=0.1, range_spec=RangeSpec(min=0, max=2, step=0.01)
        ),
        IntInput(
            name="seed",
            display_name="Seed",
            info="The seed controls the reproducibility of the job.",
            advanced=True,
            value=1,
        ),
    ]
    def fetch_models(self) -> list[str]:
        """Fetch the list of available models from GPUStack API."""
        try:
            response = requests.get(
                f"{self.gpustack_api_base}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            models_data = response.json().get("items", [])
            embedding_models = [
                model["name"]
                for model in models_data
                if "categories" in model and "llm" in model["categories"]
            ]            
            return embedding_models
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error fetching models from GPUStack: {e}")

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        """Update the build configuration with available models."""
        if field_name in ("gpustack_api_base", "model_name", "api_key") and field_value:
            try:
                models = self.fetch_models()
                if models:
                    build_config["model_name"]["options"] = models
                    build_config["model_name"]["value"] = models[0]
            except Exception as e:
                raise ValueError(f"Error getting model names: {e}") from e
        return build_config
    
    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        openai_api_key = self.api_key
        temperature = self.temperature
        model_name: str = self.model_name
        max_tokens = self.max_tokens
        model_kwargs = self.model_kwargs or {}
        openai_api_base = self.gpustack_api_base or "https://api.gpustack.com/v1"
        json_mode = self.json_mode
        seed = self.seed

        api_key = SecretStr(openai_api_key).get_secret_value() if openai_api_key else None
        output = ChatOpenAI(
            max_tokens=max_tokens or None,
            model_kwargs=model_kwargs,
            model=model_name,
            base_url=openai_api_base,
            api_key=api_key,
            temperature=temperature if temperature is not None else 0.1,
            seed=seed,
        )
        if json_mode:
            output = output.bind(response_format={"type": "json_object"})

        return output

    def _get_exception_message(self, e: Exception):
        """Get a message from an GPUStack exception.

        Args:
            e (Exception): The exception to get the message from.

        Returns:
            str: The message from the exception.
        """
        try:
            from openai import BadRequestError
        except ImportError:
            return None
        if isinstance(e, BadRequestError):
            message = e.body.get("message")
            if message:
                return message
        return None
