import asyncio 
import requests
from fastmcp import FastMCP

from src.utils.secrets import is_stdio


server = FastMCP(name='Topic Experiment')


@server.prompt()
def explain_topic(topic : str) -> str:
    "Генерирует промпт для объяснения темы в дружелюбной манере"
    return f"Can you explain {topic} in a beginner friendly manner with simple wordings and no technical jargon. Include Concept & Examples."


@server.resource("entry-weather-example://weather-letter", mime_type="text/markdown")
def write_3way_intro() -> str:
    """Пример написания приветственного сообщения, если пользователь хочет информацию по погоде"""
    with open("src/data/entry_letter.md", "r", encoding="utf-8") as file:
        return file.read()
    
@server.resource("info-excursion://Paris-trips", mime_type="text/markdown")
def write_excursion_intro() -> str:
    """Информация по экскурсиям в Париже"""
    with open("src/data/paris_trips.md", "r", encoding="utf-8") as file:
        return file.read()
    
@server.tool()
async def add(a: int, b: int) -> str:
    """Сумма двух чисел"""
    return str(a + b)


@server.tool()
def get_weather(city: str) -> str:
    """
    Получает текущую погоду в городе.
    """

    # 1 Получаем координаты города
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = requests.get(geo_url, params={"name": city, "count": 1})
    geo_data = geo_resp.json()

    if "results" not in geo_data:
        return "Город не найден."

    lat = geo_data["results"][0]["latitude"]
    lon = geo_data["results"][0]["longitude"]

    #  Получаем погоду
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_resp = requests.get(
        weather_url,
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True
        },
    )
    weather_data = weather_resp.json()

    temp = weather_data["current_weather"]["temperature"]
    wind = weather_data["current_weather"]["windspeed"]

    return f"Погода в {city}: {temp}°C, ветер {wind} км/ч"


if __name__ == "__main__":
    if is_stdio == 'True':
        # уровен протокола stdio
        server.run(transport="stdio")
    else:
        # уровен протокола streamable-http
        server.run(transport="streamable-http")