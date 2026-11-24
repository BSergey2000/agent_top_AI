import asyncio
import os
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from loguru import logger

from src.utils.secrets import OPENROUTER_API_KEY, TAVILY_API_KEY

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_PATH = PROJECT_DIR / "mcp-server.py"

WEATHER_LETTER_URI = "entry-weather-example://weather-letter"
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"

TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"
# У Tavily есть ещё crawl, map и research — они тратят много кредитов, агенту не даём
TAVILY_TOOLS = {"tavily_search", "tavily_extract"}
LOAD_ATTEMPTS = 3


def get_tools_used(response):
    """
    Извлекает информацию об инструментах, использованных в ответе агента:
    имя, аргументы вызова и результат.
    """
    # аргументы лежат в AIMessage.tool_calls, результат — в ToolMessage с тем же id
    args_by_id = {}
    for m in response["messages"]:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                args_by_id[call["id"]] = call["args"]

    tools = []
    for m in response["messages"]:
        if isinstance(m, ToolMessage):
            tools.append({
                "name": m.name,
                "args": args_by_id.get(m.tool_call_id, {}),
                "result": content_to_text(m.content),
                "status": m.status,
                "tool_call_id": m.tool_call_id,
            })
    return tools


def content_to_text(content) -> str:
    """Содержимое сообщения бывает строкой или списком блоков — приводим к строке."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict):
            parts.append(str(block.get("text", "")))
        else:
            parts.append(str(block))
    return "\n".join(part for part in parts if part)


def resources_to_dict(resources) -> dict[str, str]:
    """Превращает список ресурсов MCP в словарь {uri: текст}."""
    contents = {}
    for blob in resources:
        uri = str(blob.metadata["uri"])
        logger.info(f"URI: {uri}, MIME type: {blob.mimetype}")
        contents[uri] = blob.as_string()
    return contents


def get_resource_content(resources, uri: str = WEATHER_LETTER_URI):
    """
    Извлекает содержимое ресурса с определенным URI из списка ресурсов.
    """
    return resources_to_dict(resources).get(uri)


def web_tools_hint(tool_names) -> str:
    """Подсказка про инструменты интернета — по тем именам, которые реально доступны агенту."""
    names = set(tool_names)
    if "tavily_search" in names:
        return ("Если нужна актуальная информация из интернета, используй инструмент tavily_search, "
                "а чтобы прочитать страницу по адресу - tavily_extract. ")
    if "search" in names:
        return ("Если нужна актуальная информация из интернета, используй инструмент search, "
                "а чтобы прочитать найденную страницу - fetch_content. ")
    return ""


def build_system_prompt(resources: dict[str, str], style: str = "", tool_names=()) -> str:
    """
    Собирает system_prompt агента: роль, правила работы с инструментами
    и большой контекст из переданных ресурсов MCP.
    """
    parts = [
        "Ты - ассистент, который может использовать инструменты для выполнения задач. "
        "Отвечай на русском языке. Для арифметики вызывай add, для погоды - get_weather, "
        "для перевода температуры между шкалами - convert_temperature. "
        + web_tools_hint(tool_names) +
        "Если инструмент вернул ошибку, честно сообщи о ней пользователю и не выдумывай данные."
    ]
    if style:
        parts.append(style)

    letter = resources.get(WEATHER_LETTER_URI)
    if letter:
        parts.append(
            "Если использован инструмент по погоде get_weather, начни ответ с приветствия "
            f"по этому образцу, продолжив его данными о погоде: {letter.strip()}"
        )

    context = {uri: text for uri, text in resources.items() if uri != WEATHER_LETTER_URI}
    if context:
        parts.append(
            "Справочные материалы. Если вопрос касается этих тем, отвечай строго по ним; "
            "если нужных сведений в материалах нет - так и скажи."
        )
        for uri, text in context.items():
            parts.append(f'<resource uri="{uri}">\n{text.strip()}\n</resource>')

    return "\n\n".join(parts)


def web_search_server() -> dict:
    """
    Внешний MCP сервер поиска в интернете.
    С ключом TAVILY_API_KEY - удалённый сервер Tavily по streamable-http
    (инструменты tavily_search и tavily_extract). Без ключа - локальный сервер
    DuckDuckGo по stdio (search и fetch_content); он не требует ключа, но
    DuckDuckGo часто блокирует программные запросы.
    """
    if TAVILY_API_KEY:
        return {
            "transport": "streamable_http",
            "url": TAVILY_MCP_URL,
            "headers": {"Authorization": f"Bearer {TAVILY_API_KEY}"},
        }
    return {
        "transport": "stdio",
        "command": "uvx",
        "args": ["duckduckgo-mcp-server"],
    }


def client_setup():
    client = MultiServerMCPClient(
        {
            "main": {
                "transport": "stdio",
                # тот же интерпретатор, что и у агента: в нём уже стоят зависимости сервера
                "command": sys.executable,
                "args": [str(SERVER_PATH)],
                "cwd": str(PROJECT_DIR),
                # агент общается с сервером через stdio, независимо от IS_STDIO в .env
                "env": {**os.environ, "IS_STDIO": "True"},
            },
            "web_search": web_search_server(),
        }
    )
    return client


async def load_tools(client):
    """
    Загружает инструменты со всех серверов. Внешний сервер поиска необязателен:
    если он не запустился, агент продолжает работать с инструментами своего сервера.
    """
    tools = await client.get_tools(server_name="main")
    for attempt in range(1, LOAD_ATTEMPTS + 1):
        try:
            web_tools = await client.get_tools(server_name="web_search")
            break
        except Exception as error:
            # удалённый сервер может не ответить с первого раза - пробуем ещё
            logger.warning(f"Сервер поиска web_search не ответил (попытка {attempt}): {error!r}")
            await asyncio.sleep(2)
    else:
        logger.warning("Сервер поиска web_search недоступен, работаем без него")
        return tools
    if TAVILY_API_KEY:
        web_tools = [tool for tool in web_tools if tool.name in TAVILY_TOOLS]
    return tools + web_tools


def model_setup():
    model = ChatOpenAI(
        # модель можно сменить через OPENROUTER_MODEL в .env, не трогая код
        model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        api_key=OPENROUTER_API_KEY,
        # с запасом: модель тратит часть лимита на скрытые рассуждения, и при малом
        # значении ответ обрывается или приходит пустым (finish_reason=length)
        max_tokens=4000,
        timeout=90,
        max_retries=4)
    return model
