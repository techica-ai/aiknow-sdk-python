"""
Asynchronous SDK usage example.

Set environment variables before running:
    AIKNOW_API_KEY=your-api-key
    AIKNOW_BASE_URL=http://localhost:8000/api/v1  (optional)

Run:
    python examples/async_example.py
"""
from __future__ import annotations

import asyncio

from aiknow import AIKnowAPIError, AIKnowConnectionError, AsyncAIKnowClient


async def main() -> None:
    print("=== AIKNOW SDK: Async Example ===")

    async with AsyncAIKnowClient() as client:
        if not await client.ping():
            print("❌ API Server is not responding. Please start the server first.")
            return
        print("✅ Server is up.\n")

        print("Chatting with agent (non-blocking)...")
        try:
            chat_res = await client.chat.ask(
                query="Hello, who are you?",
                tenant_id="tenant-demo",
            )
            print(f"   ✅ Answer: {chat_res.answer}")
            if chat_res.trace_id:
                print(f"   🔍 Trace ID: {chat_res.trace_id}")
        except (AIKnowAPIError, AIKnowConnectionError) as e:
            print(f"   ❌ Chat error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
