from __future__ import annotations

from typing import Any, cast

import httpx
import yaml
from aiknow_contracts.graph_models import ConversationGraphConfig, GraphType

from .._http import raise_for_status, wrap_httpx_errors


class _GraphsResourceBase:
    pass

class GraphsResource(_GraphsResourceBase):
    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def register(
        self,
        config: ConversationGraphConfig | dict[str, Any],
        graph_type: GraphType = GraphType.CONVERSATION_GRAPH,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        if isinstance(config, dict):
            config = ConversationGraphConfig.model_validate(config)
        payload = config.model_dump()
        try:
            res = self._client.post(
                "/graphs",
                json=payload,
                params={"graph_type": graph_type.value, "tenant_id": tenant_id},
            )
        except Exception as exc:
            wrap_httpx_errors("Graphs.register", exc)
        raise_for_status("Graphs.register", res)
        return cast(dict[str, Any], res.json())

    def register_from_yaml(
        self,
        file_path: str,
        graph_type: GraphType = GraphType.CONVERSATION_GRAPH,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return self.register(config=data, graph_type=graph_type, tenant_id=tenant_id)


class AsyncGraphsResource(_GraphsResourceBase):
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def register(
        self,
        config: ConversationGraphConfig | dict[str, Any],
        graph_type: GraphType = GraphType.CONVERSATION_GRAPH,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        if isinstance(config, dict):
            config = ConversationGraphConfig.model_validate(config)
        payload = config.model_dump()
        try:
            res = await self._client.post(
                "/graphs",
                json=payload,
                params={"graph_type": graph_type.value, "tenant_id": tenant_id},
            )
        except Exception as exc:
            wrap_httpx_errors("Graphs.register", exc)
        raise_for_status("Graphs.register", res)
        return cast(dict[str, Any], res.json())

    async def register_from_yaml(
        self,
        file_path: str,
        graph_type: GraphType = GraphType.CONVERSATION_GRAPH,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return await self.register(config=data, graph_type=graph_type, tenant_id=tenant_id)
