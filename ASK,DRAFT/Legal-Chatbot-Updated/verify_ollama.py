import requests
import os
from langchain_ollama import ChatOllama

def verify_connection():
    ip = '192.168.0.25'
    url = f'http://{ip}:11434'
    
    print(f"--- NETWORK DIAGNOSTICS ---")
    print(f"1. Testing raw requests to {url}/api/tags...")
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        print(f"   ✅ Success! Status: {r.status_code}")
    except Exception as e:
        print(f"   ❌ Raw Request Failed: {e}")
        return

    print(f"\n2. Testing ChatOllama with NO_PROXY...")
    os.environ['NO_PROXY'] = ip
    llm = ChatOllama(model='qwen2.5:14b', base_url=url, timeout=10)
    try:
        # Simple test
        res = llm.invoke("Hi")
        print(f"   ✅ Success! Response: {res.content}")
    except Exception as e:
        print(f"   ❌ ChatOllama Failed: {e}")
        print("\nPossible solutions:")
        print("a) Check if a VPN is running and try turning it off.")
        print("b) Ensure the Ollama machine has OLLAMA_HOST=0.0.0.0 set.")

if __name__ == "__main__":
    verify_connection()
