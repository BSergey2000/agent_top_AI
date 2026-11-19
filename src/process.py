from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from loguru import logger

from src.utils.secrets import OPENROUTER_API_KEY

def get_tools_used(response):
    """
    Извлекает информацию об инструментах, использованных в ответе агента.
    """
    tools = []
    for m in response["messages"]:
        if isinstance(m, ToolMessage):
            tools.append({
                "name": m.name,         
                "result": m.content,   
                "tool_call_id": m.tool_call_id 
            })
    return tools

def get_resource_content(resources):
    """
    Извлекает содержимое ресурса с определенным URI из списка ресурсов.
    """
    for blob in resources:
        logger.info(f"URI: {blob.metadata['uri']}, MIME type: {blob.mimetype}")
        if str(blob.metadata['uri']) in "entry-weather-example://weather-letter":
            content = blob.as_string()
            return content
        
def client_setup():
    client = MultiServerMCPClient(
        {
            "main": {
                "transport": "stdio", 
                "command": "python",
                "args": ["C:\\documents\\проект_УРФУ\\программа урфу\\agent_top_AI\\mcp-server.py"],
                # агент общается с сервером через stdio, независимо от IS_STDIO в .env
                "env": {"IS_STDIO": "True"},
            },
            # внешний MCP сервер поиска в интернете (DuckDuckGo, API-ключ не нужен)
            # инструменты: search(query, max_results) и fetch_content(url)
            "web_search": {
                "transport": "stdio",
                "command": "uvx",
                "args": ["duckduckgo-mcp-server"],
            },
        }
    )
    return client

def model_setup():
    model = ChatOpenAI(
        model="nvidia/nemotron-3.5-super-vl:free",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        api_key=OPENROUTER_API_KEY,
        max_tokens=1000,
        timeout=30)
    return model