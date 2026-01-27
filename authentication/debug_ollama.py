from langchain_openai import ChatOpenAI
import os
import traceback

os.environ['NO_PROXY'] = '192.168.0.25'

def test_ollama():
    url = 'http://192.168.0.25:11434/v1'
    model = 'qwen2.5:latest' # Using latest as we know it exists
    print(f"DEBUG: Testing ChatOpenAI at {url} with model {model}")
    
    llm = ChatOpenAI(
        model=model,
        base_url=url,
        api_key='ollama', # Required but ignored
        timeout=30
    )
    
    try:
        response = llm.invoke("Hi")
        print("SUCCESS:", response.content)
    except Exception as e:
        print("FAILURE:")
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama()
