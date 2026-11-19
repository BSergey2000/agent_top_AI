import asyncio
from langchain.agents import create_agent
from loguru import logger

from src.process import get_tools_used, get_resource_content, model_setup, client_setup

async def main():
    # Инициализация клиента и модели
    client = client_setup()
    model = model_setup()
    
    # Используем инструменты, предоставленные сервером
    tools = await client.get_tools()
    logger.info(f"Доступные инструменты: {tools}")
    
    # Используем ресурсы для доп контекста
    # ресурсы есть только у нашего сервера, поэтому берём их только из "main"
    resources = await client.get_resources("main")
    logger.info(f"Доступные ресурсы: {resources}")
    
    weather_prompt = get_resource_content(resources)
    logger.info(f"Приветствие для инструмента погоды: {weather_prompt}\n")
    
    agent = create_agent(
        model=model,
        tools = tools,
        system_prompt=f'Ты - ассистент, который может использовать инструменты для выполнения задач. Если выбран инструмент по погоде - get_weather, то используй приветствие: {weather_prompt}. Если нужна актуальная информация из интернета, используй инструмент search, а чтобы прочитать найденную страницу - fetch_content.',
    )
    math_response = await agent.ainvoke(
    {"messages": [("user", "сколько будет 2 + 2?")]}
)
    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Какая сегодня погода в Yekaterinburg?"}]}
    )
    search_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Найди в интернете, какие последние новости про Model Context Protocol"}]}
    )
    logger.info(f'Ответ по математике: {math_response["messages"][-1].content}\n')
    logger.info(f'Использованные инструменты: {get_tools_used(math_response)}\n')
    
    logger.info(f'Ответ по погоде: {weather_response["messages"][-1].content}\n')
    logger.info(f'Использованные инструменты: {get_tools_used(weather_response)}\n')

    logger.info(f'Ответ из интернета: {search_response["messages"][-1].content}\n')
    logger.info(f'Использованные инструменты: {get_tools_used(search_response)}\n')

if __name__ == "__main__":
    asyncio.run(main())