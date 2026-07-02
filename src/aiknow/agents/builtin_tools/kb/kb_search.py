"""KB tools: kb_search, kb_list_sources, kb_get_document."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiknow.agents.builtin_tools._base import BuiltinTool, ToolParam

if TYPE_CHECKING:
    from aiknow.agents.builtin_tools._toolbox import BuiltinToolContext


class KbSearchTool(BuiltinTool):
    """Hybrid vector search across Knowledge Bases.

    Performs dense + sparse retrieval with optional reranking.
    Returns top-k chunks with source metadata.
    """

    name = "kb_search"
    description = {
        "vi": "Tìm kiếm thông tin trong Knowledge Base bằng hybrid search",
        "en": "Search the knowledge base using hybrid vector search",
    }
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description={"vi": "Câu truy vấn tìm kiếm", "en": "Search query"},
            required=True,
        ),
        ToolParam(
            name="knowledge_bases",
            type="array",
            description={"vi": "Danh sách KB ID cần tìm", "en": "List of KB IDs to search"},
            required=False,
            default=None,
        ),
        ToolParam(
            name="top_k",
            type="integer",
            description={"en": "Number of chunks to return"},
            required=False,
            default=5,
        ),
        ToolParam(
            name="rerank",
            type="boolean",
            description={"en": "Whether to apply reranking"},
            required=False,
            default=True,
        ),
    ]

    async def execute(self, ctx: BuiltinToolContext, **kwargs: Any) -> str:
        query: str = kwargs.get("query", "")
        top_k: int = int(kwargs.get("top_k", 5))
        rerank: bool = bool(kwargs.get("rerank", True))
        kb_ids: list[str] | None = kwargs.get("knowledge_bases")

        if not query:
            return "[kb_search] Missing required parameter: query"

        if ctx.http_client is None:
            # No HTTP client available — return a helpful message
            return f"[kb_search] No results available (HTTP client not configured). Query: {query}"

        try:
            params: dict[str, Any] = {
                "query": query,
                "top_k": top_k,
                "rerank": rerank,
                "tenant_id": ctx.tenant_id,
            }
            if kb_ids:
                params["knowledge_base_ids"] = kb_ids

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if ctx.api_key:
                headers["Authorization"] = f"Bearer {ctx.api_key}"

            response = await ctx.http_client.post(
                f"kb/search",
                json=params,
                headers=headers,
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            chunks = data.get("results", [])

            if not chunks:
                return f"Không tìm thấy thông tin liên quan đến: {query}"

            lines = [f"Kết quả tìm kiếm cho: **{query}**\n"]
            for i, chunk in enumerate(chunks[:top_k], 1):
                content = chunk.get("content", "")
                source = chunk.get("source", {})
                score = chunk.get("score", 0.0)
                doc_name = source.get("document_name", "Unknown")
                lines.append(f"**[{i}]** (score: {score:.2f}) — *{doc_name}*")
                lines.append(content)
                lines.append("")

            return "\n".join(lines)

        except Exception as exc:  # noqa: BLE001
            return f"[kb_search] Error: {exc}"


class KbListSourcesTool(BuiltinTool):
    """List all available Knowledge Base sources for the tenant."""

    name = "kb_list_sources"
    description = {
        "vi": "Liệt kê các nguồn dữ liệu trong Knowledge Base",
        "en": "List available knowledge base sources",
    }
    parameters = [
        ToolParam(
            name="knowledge_base_id",
            type="string",
            description={"en": "KB ID to list sources from (optional — lists all if omitted)"},
            required=False,
        ),
    ]

    async def execute(self, ctx: BuiltinToolContext, **kwargs: Any) -> str:
        kb_id: str = kwargs.get("knowledge_base_id", "")

        if ctx.http_client is None:
            return "[kb_list_sources] HTTP client not configured."

        try:
            url = f"kb"
            if kb_id:
                url += f"/{kb_id}/sources"
            headers: dict[str, str] = {}
            if ctx.api_key:
                headers["Authorization"] = f"Bearer {ctx.api_key}"

            response = await ctx.http_client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            items = data.get("items", data) if isinstance(data, dict) else data

            if not items:
                return "Không có Knowledge Base nào được cấu hình."

            lines = ["**Danh sách Knowledge Base:**\n"]
            for item in items:
                name = item.get("name", item.get("id", "Unknown"))
                doc_count = item.get("document_count", "?")
                lines.append(f"- **{name}** ({doc_count} tài liệu)")

            return "\n".join(lines)

        except Exception as exc:  # noqa: BLE001
            return f"[kb_list_sources] Error: {exc}"


class KbGetDocumentTool(BuiltinTool):
    """Retrieve a specific document from the Knowledge Base by ID."""

    name = "kb_get_document"
    description = {
        "vi": "Lấy nội dung đầy đủ của một tài liệu từ Knowledge Base",
        "en": "Get full content of a specific knowledge base document",
    }
    parameters = [
        ToolParam(
            name="document_id",
            type="string",
            description={"en": "Document ID to retrieve"},
            required=True,
        ),
        ToolParam(
            name="knowledge_base_id",
            type="string",
            description={"en": "KB ID containing the document"},
            required=True,
        ),
    ]

    async def execute(self, ctx: BuiltinToolContext, **kwargs: Any) -> str:
        doc_id: str = kwargs.get("document_id", "")
        kb_id: str = kwargs.get("knowledge_base_id", "")

        if not doc_id or not kb_id:
            return "[kb_get_document] Missing required parameters: document_id, knowledge_base_id"

        if ctx.http_client is None:
            return "[kb_get_document] HTTP client not configured."

        try:
            headers: dict[str, str] = {}
            if ctx.api_key:
                headers["Authorization"] = f"Bearer {ctx.api_key}"

            response = await ctx.http_client.get(
                f"kb/{kb_id}/documents/{doc_id}",
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("content", "")
            title = data.get("title", doc_id)
            return f"**{title}**\n\n{content}"

        except Exception as exc:  # noqa: BLE001
            return f"[kb_get_document] Error: {exc}"
