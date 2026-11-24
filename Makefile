# Установка зависимостей
setup:
	uv sync

# Запуск MCP сервера STDIO
mcp_stdio:
	npx @modelcontextprotocol/inspector uv run --with fastmcp mcp-server.py

# Запуск MCP сервера Streamable HTTP, необходимо выполнить две команды: mcp, inspector
mcp:
	uv run --with fastmcp --with requests --with starlette --with uvicorn mcp-server.py

# Запуск MCP Inspector
inspector:
	npx @modelcontextprotocol/inspector

# Запуск агента, который будет взаимодействовать с MCP сервером (эксперименты задания 8)
agent:
	uv run python main.py

# Эксперименты с сервером: prompt, resources, tools и оба транспорта (задания 1-5)
experiments:
	uv run python experiments_server.py
