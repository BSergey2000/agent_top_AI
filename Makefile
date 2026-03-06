# Запуск MCP сервера STDIO
mcp_stdio:
	npx @modelcontextprotocol/inspector uv run --with fastmcp mcp-server.py

# Запуск MCP сервера Streamable HTTP, необходимо выполнить две команды: mcp, inspector
mcp:
	uv run --with fastmcp --with requests --with starlette --with uvicorn mcp-server.py

# Запуск MCP Inspector
inspector:
	npx @modelcontextprotocol/inspector

# Запустить всё (в двух терминалах лучше)
run:
	@echo "Run 'make mcp' in one terminal and 'make inspector' in another."