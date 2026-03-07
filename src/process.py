from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from loguru import logger

from utils.secrets import OPEN_AI_KEY

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
            }
        }
    )
    return client

def model_setup():
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=OPEN_AI_KEY,
        max_tokens=1000,
        timeout=30)
    return model