import os
from dotenv import load_dotenv

class EnvironmentVariables:
    load_dotenv()
    APP_ENV = os.getenv("APP_ENV", "local")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.0.56:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
