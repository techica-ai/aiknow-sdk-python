"""
Synchronous SDK usage example.

Set environment variables before running:
    AIKNOW_API_KEY=your-api-key
    AIKNOW_BASE_URL=http://localhost:8000/api/v1  (optional)

Run:
    python examples/sync_example.py
"""
from __future__ import annotations

import os
import tempfile

from aiknow import AIKnowAPIError, AIKnowClient, AIKnowConnectionError


def main() -> None:
    print("=== AIKNOW SDK: Sync Example ===")

    with AIKnowClient() as client:
        if not client.ping():
            print("❌ API Server is not responding. Please start the server first.")
            return
        print("✅ Server is up.\n")

        # 1. Upload a document
        print("1. Uploading document...")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("AIKNOW User Guide. Author: OSP Team.")
            tmp_path = tmp.name

        try:
            upload_res = client.ingestion.upload(tmp_path, tenant_id="tenant-demo")
            print(f"   ✅ Upload successful. Job ID: {upload_res.job_id}")
        except (AIKnowAPIError, AIKnowConnectionError) as e:
            print(f"   ❌ Upload error: {e}")
        finally:
            os.unlink(tmp_path)

        # 2. Chat with the agent
        print("\n2. Chatting with agent...")
        try:
            chat_res = client.chat.ask(
                query="Who wrote the AIKNOW user guide?",
                tenant_id="tenant-demo",
            )
            print(f"   ✅ Answer: {chat_res.answer}")
            if chat_res.trace_id:
                print(f"   🔍 Trace ID: {chat_res.trace_id}")
        except (AIKnowAPIError, AIKnowConnectionError) as e:
            print(f"   ❌ Chat error: {e}")


if __name__ == "__main__":
    main()
