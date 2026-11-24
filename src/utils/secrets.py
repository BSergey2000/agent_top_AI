import os
from dotenv import load_dotenv

load_dotenv()  

is_stdio = os.getenv("IS_STDIO")
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = (os.getenv("TAVILY_API_KEY") or "").strip()
