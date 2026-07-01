"""
AsyncPromptConfigResource — SDK resource for prompt configs.

Maps to the /api/v1/prompt-configs/* endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import httpx
from aiknow_contracts.prompt_config import PromptConfig, PromptConfigCreate, PromptConfigList

logger = logging.getLogger(__name__)

_BASE_PATH = "/api/v1/prompt-configs/"


class AsyncPromptConfigResource:
    """SDK resource for managing prompt configurations.

    Usage::

        async with AsyncAIKnowClient(...) as client:
            # Register a new config
            await client.prompt_configs.register(
                name="my_config",
                type="extraction",
                system_prompt="Extract names..."
            )

            # List all configs
            configs = await client.prompt_configs.list()

            # Get a specific config
            config = await client.prompt_configs.get("my_config")

            # Update a config
            await client.prompt_configs.update("my_config", system_prompt="Updated prompt...")

            # Delete a config
            await client.prompt_configs.delete("my_config")
    """

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def register(
        self,
        name: str,
        type: Literal["extraction", "classification", "evaluation", "generation"],
        system_prompt: str,
        user_prompt_template: str | None = None,
        output_schema: dict[str, Any] | None = None,
        model: str | None = None,
        description: str | None = None,
    ) -> str:
        """Register a new prompt config.

        Args:
            name: Unique name for the config within the tenant.
            type: Type of prompt.
            system_prompt: System prompt template.
            user_prompt_template: Optional user prompt template.
            output_schema: Optional JSON schema for output.
            model: Optional LLM model override.
            description: Optional description.

        Returns:
            The name of the created config.
        """
        payload = PromptConfigCreate(
            name=name,
            type=type,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            output_schema=output_schema,
            model=model,
            description=description,
        )

        resp = await self._http.post(_BASE_PATH, json=payload.model_dump(exclude_none=True))
        resp.raise_for_status()
        return resp.json().get("name")

    async def list(self) -> PromptConfigList:
        """List all active prompt configs.

        Returns:
            PromptConfigList containing the configs.
        """
        resp = await self._http.get(_BASE_PATH)
        resp.raise_for_status()
        return PromptConfigList.model_validate(resp.json())

    async def get(self, name: str) -> PromptConfig:
        """Get a specific prompt config by name.

        Args:
            name: The name of the config to get.

        Returns:
            PromptConfig object.
        """
        resp = await self._http.get(f"{_BASE_PATH}/{name}")
        resp.raise_for_status()
        return PromptConfig.model_validate(resp.json())

    async def update(
        self,
        name: str,
        type: Literal["extraction", "classification", "evaluation", "generation"],
        system_prompt: str,
        user_prompt_template: str | None = None,
        output_schema: dict[str, Any] | None = None,
        model: str | None = None,
        description: str | None = None,
    ) -> str:
        """Update an existing prompt config.

        Args:
            name: Name of the config to update.
            type: Type of prompt.
            system_prompt: System prompt template.
            user_prompt_template: Optional user prompt template.
            output_schema: Optional JSON schema for output.
            model: Optional LLM model override.
            description: Optional description.

        Returns:
            The name of the updated config.
        """
        payload = PromptConfigCreate(
            name=name,
            type=type,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            output_schema=output_schema,
            model=model,
            description=description,
        )

        resp = await self._http.put(
            f"{_BASE_PATH}/{name}", json=payload.model_dump(exclude_none=True)
        )
        resp.raise_for_status()
        return resp.json().get("name")

    async def delete(self, name: str) -> bool:
        """Deactivate a prompt config.

        Args:
            name: The name of the config to deactivate.

        Returns:
            True if deactivated.
        """
        resp = await self._http.delete(f"{_BASE_PATH}/{name}")
        resp.raise_for_status()
        return resp.json().get("deleted", False)


class PromptConfigResource:
    """SDK resource for managing prompt configurations (synchronous)."""

    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def register(
        self,
        name: str,
        type: Literal["extraction", "classification", "evaluation", "generation"],
        system_prompt: str,
        user_prompt_template: str | None = None,
        output_schema: dict[str, Any] | None = None,
        model: str | None = None,
        description: str | None = None,
    ) -> str:
        payload = PromptConfigCreate(
            name=name,
            type=type,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            output_schema=output_schema,
            model=model,
            description=description,
        )
        resp = self._http.post(_BASE_PATH, json=payload.model_dump(exclude_none=True))
        resp.raise_for_status()
        return resp.json().get("name")

    def list(self) -> PromptConfigList:
        resp = self._http.get(_BASE_PATH)
        resp.raise_for_status()
        return PromptConfigList.model_validate(resp.json())

    def get(self, name: str) -> PromptConfig:
        resp = self._http.get(f"{_BASE_PATH}/{name}")
        resp.raise_for_status()
        return PromptConfig.model_validate(resp.json())

    def update(
        self,
        name: str,
        type: Literal["extraction", "classification", "evaluation", "generation"],
        system_prompt: str,
        user_prompt_template: str | None = None,
        output_schema: dict[str, Any] | None = None,
        model: str | None = None,
        description: str | None = None,
    ) -> str:
        payload = PromptConfigCreate(
            name=name,
            type=type,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            output_schema=output_schema,
            model=model,
            description=description,
        )
        resp = self._http.put(f"{_BASE_PATH}/{name}", json=payload.model_dump(exclude_none=True))
        resp.raise_for_status()
        return resp.json().get("name")

    def delete(self, name: str) -> bool:
        resp = self._http.delete(f"{_BASE_PATH}/{name}")
        resp.raise_for_status()
        return resp.json().get("deleted", False)
