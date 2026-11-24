"""
Эксперименты с MCP сервером (задания 1-5): prompt, resources, tools и оба транспорта.

    python experiments_server.py            # всё, включая запросы к модели
    python experiments_server.py --no-llm   # только сервер, без запросов к модели

Результаты сохраняются в results/server_experiments.json.
"""
import asyncio
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from loguru import logger

from src.process import content_to_text, model_setup

PROJECT_DIR = Path(__file__).resolve().parent
SERVER_PATH = PROJECT_DIR / "mcp-server.py"
RESULTS_PATH = PROJECT_DIR / "results" / "server_experiments.json"
SERVER_LOG_PATH = PROJECT_DIR / "logs" / "server_stderr.log"
# свой порт, чтобы не мешать серверу, уже запущенному через make mcp на 8000
HTTP_HOST, HTTP_PORT = "127.0.0.1", 8765
HTTP_URL = f"http://{HTTP_HOST}:{HTTP_PORT}/mcp"
LATENCY_CALLS = 30

# Исходные тексты до изменений — чтобы сравнить ответы модели «до» и «после»
OLD_PROMPT = ("Can you explain {topic} in a beginner friendly manner with simple wordings "
              "and no technical jargon. Include Concept & Examples.")
OLD_PARIS_TRIPS = "Экскурсия 1: Лувр"

PROMPT_TOPICS = [
    ("простая тема", "радуга"),
    ("сложная тема", "квантовая запутанность"),
    ("некорректные данные: набор символов", "asdfgh 12345 ###"),
    ("некорректные данные: пустая строка", ""),
    ("некорректные данные: 500 символов", "а" * 500),
]

TOOL_CALLS = [
    ("add", {"a": 2, "b": 2}),
    ("add", {"a": 123456789012345678901234567890, "b": 1}),
    ("add", {"a": -5, "b": -7}),
    ("add", {"a": "3", "b": "4"}),
    ("add", {"a": "abc", "b": 1}),
    ("add", {"a": 1.5, "b": 2}),
    ("add", {"a": 1}),
    ("get_weather", {"city": "Paris"}),
    ("get_weather", {"city": "PARIS"}),
    ("get_weather", {"city": "paris"}),
    ("get_weather", {"city": "Екатеринбург"}),
    ("get_weather", {"city": "Tokyo"}),
    ("get_weather", {"city": "Qwertyzzzxx"}),
    ("get_weather", {"city": ""}),
    ("get_weather", {"city": 12345}),
    ("convert_temperature", {"value": 100, "from_unit": "C", "to_unit": "F"}),
    ("convert_temperature", {"value": -40, "from_unit": "F", "to_unit": "C"}),
    ("convert_temperature", {"value": 0, "from_unit": "kelvin", "to_unit": "°C"}),
    ("convert_temperature", {"value": -300, "from_unit": "C", "to_unit": "K"}),
    ("convert_temperature", {"value": 10, "from_unit": "X", "to_unit": "C"}),
    ("convert_temperature", {"value": "жарко", "from_unit": "C", "to_unit": "F"}),
]


def stdio_client() -> Client:
    transport = StdioTransport(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        env={**os.environ, "IS_STDIO": "True"},
        cwd=str(PROJECT_DIR),
        # баннер и трассировки сервера пишем в файл, чтобы не засорять вывод экспериментов
        log_file=SERVER_LOG_PATH,
    )
    return Client(transport)


def start_http_server() -> subprocess.Popen:
    """Запускает сервер в режиме streamable-http отдельным процессом и ждёт, пока откроется порт."""
    process = subprocess.Popen(
        [sys.executable, str(SERVER_PATH)],
        env={**os.environ, "IS_STDIO": "False", "FASTMCP_PORT": str(HTTP_PORT)},
        cwd=str(PROJECT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"HTTP сервер завершился сразу после запуска (занят порт {HTTP_PORT}?)")
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex((HTTP_HOST, HTTP_PORT)) == 0:
                return process
        time.sleep(0.3)
    process.kill()
    raise RuntimeError(f"HTTP сервер не открыл порт {HTTP_PORT} за 30 секунд")


async def call_tool(client: Client, name: str, args: dict) -> dict:
    started = time.perf_counter()
    try:
        result = await client.call_tool(name, args)
        output, ok = content_to_text([block.text for block in result.content]), True
    except Exception as error:
        output, ok = str(error), False
    return {"tool": name, "args": args, "ok": ok, "output": output,
            "ms": round((time.perf_counter() - started) * 1000, 1)}


async def get_prompt(client: Client, topic: str) -> dict:
    try:
        result = await client.get_prompt("explain_topic", {"topic": topic})
        return {"ok": True, "text": result.messages[0].content.text}
    except Exception as error:
        return {"ok": False, "text": str(error)}


async def measure_transport(make_client) -> dict:
    """Время подключения (с initialize) и задержка вызова инструмента add на готовом соединении."""
    started = time.perf_counter()
    async with make_client() as client:
        connect_ms = (time.perf_counter() - started) * 1000
        tools = await client.list_tools()
        samples = []
        for i in range(LATENCY_CALLS):
            call_started = time.perf_counter()
            await client.call_tool("add", {"a": i, "b": 1})
            samples.append((time.perf_counter() - call_started) * 1000)
    return {
        "connect_ms": round(connect_ms, 1),
        "call_median_ms": round(statistics.median(samples), 2),
        "call_min_ms": round(min(samples), 2),
        "call_max_ms": round(max(samples), 2),
        "calls": LATENCY_CALLS,
        "tools": [tool.name for tool in tools],
    }


async def ask_model(model, text: str, system: str = "") -> str:
    messages = ([("system", system)] if system else []) + [("user", text)]
    try:
        response = await model.ainvoke(messages)
        return content_to_text(response.content)
    except Exception as error:
        return f"ОШИБКА: {error!r}"


async def main():
    use_llm = "--no-llm" not in sys.argv
    SERVER_LOG_PATH.parent.mkdir(exist_ok=True)
    results = {}

    async with stdio_client() as client:
        # --- что вообще есть на сервере
        results["inventory"] = {
            "tools": [{"name": t.name, "description": (t.description or "").strip()}
                      for t in await client.list_tools()],
            "prompts": [p.name for p in await client.list_prompts()],
            "resources": [str(r.uri) for r in await client.list_resources()],
        }
        logger.info(f"Сервер предоставляет: {results['inventory']}")

        # --- 1. Prompt
        results["prompts"] = []
        for label, topic in PROMPT_TOPICS:
            entry = {"label": label, "topic": topic, **await get_prompt(client, topic)}
            results["prompts"].append(entry)
            logger.info(f"prompt [{label}]: ok={entry['ok']} {entry['text'][:120]!r}")

        # --- 2. Resources
        results["resources"] = {}
        for uri in results["inventory"]["resources"]:
            content = await client.read_resource(uri)
            results["resources"][uri] = content[0].text
            logger.info(f"resource {uri}: {len(content[0].text)} символов")
        try:
            await client.read_resource("info-excursion://Tokyo-trips")
            results["resource_missing"] = "ресурс неожиданно найден"
        except Exception as error:
            results["resource_missing"] = str(error)
        logger.info(f"несуществующий ресурс: {results['resource_missing']}")

        # --- 3. Tools
        results["tools"] = []
        for name, args in TOOL_CALLS:
            entry = await call_tool(client, name, args)
            results["tools"].append(entry)
            logger.info(f"tool {name}({args}): ok={entry['ok']} {entry['output'][:150]!r}")

    # --- 4. Транспорты: одинаковые замеры для stdio и streamable-http
    results["transports"] = {"stdio": await measure_transport(stdio_client)}
    http_server = start_http_server()
    try:
        results["transports"]["streamable-http"] = await measure_transport(lambda: Client(HTTP_URL))
        # к HTTP серверу можно подключиться несколькими клиентами сразу — stdio так не умеет
        async with Client(HTTP_URL) as first, Client(HTTP_URL) as second:
            answers = await asyncio.gather(
                first.call_tool("add", {"a": 1, "b": 1}),
                second.call_tool("add", {"a": 2, "b": 2}),
            )
            results["transports"]["http_two_clients"] = [a.content[0].text for a in answers]
    finally:
        http_server.kill()
    logger.info(f"транспорты: {results['transports']}")

    # --- Ответы модели на prompt и ресурсы «до» и «после» изменений
    if use_llm:
        model = model_setup()
        results["prompt_answers"] = []
        for entry in results["prompts"]:
            if not entry["ok"]:
                continue
            old_text = OLD_PROMPT.format(topic=entry["topic"])
            results["prompt_answers"].append({
                "label": entry["label"],
                "topic": entry["topic"],
                "old_prompt": old_text,
                "old_answer": await ask_model(model, old_text),
                "new_prompt": entry["text"],
                "new_answer": await ask_model(model, entry["text"]),
            })
            logger.info(f"модель ответила на prompt [{entry['label']}]")

        question = "Какие экскурсии есть в Париже и сколько они стоят?"
        system = ("Отвечай только по справочному материалу. "
                  "Если сведений в нём нет - так и скажи.\n\n{context}")
        new_paris = results["resources"]["info-excursion://Paris-trips"]
        results["resource_answers"] = {
            "question": question,
            "old_resource": OLD_PARIS_TRIPS,
            "old_answer": await ask_model(model, question, system.format(context=OLD_PARIS_TRIPS)),
            "new_resource": new_paris,
            "new_answer": await ask_model(model, question, system.format(context=new_paris)),
        }
        logger.info("модель ответила по ресурсу до и после изменения")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Результаты сохранены в {RESULTS_PATH}")


if __name__ == "__main__":
    sys.stderr.reconfigure(errors="replace")
    asyncio.run(main())
