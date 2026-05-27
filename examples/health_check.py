import os
import sys

# Đảm bảo có thể import SDK khi chạy local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aiknow import AIKnowClient

def main():
    print("=== AIKNOW SDK: Basic Health Check ===")
    
    # 1. Khởi tạo Client đồng bộ (Sync Client)
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key")
    
    with AIKnowClient(base_url="http://localhost:8000/api/v1", api_key=api_key) as client:
        # 2. Gọi hàm ping để kiểm tra kết nối
        is_connected = client.ping()
        
        if is_connected:
            print("[+] Connection Successful! API Server is running.")
        else:
            print("[-] Connection Failed! Make sure the API server is running on http://localhost:8000")
            print("    Lệnh bật server: uv run python apps/aiknow-api/src/aiknow_api/main.py")

if __name__ == "__main__":
    main()
