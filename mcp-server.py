from pathlib import Path

import requests
from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError, ToolError

from src.utils.secrets import is_stdio

# Пути считаем от файла сервера, чтобы ресурсы находились при запуске из любой папки
DATA_DIR = Path(__file__).resolve().parent / "src" / "data"
REQUEST_TIMEOUT = 10  # секунд на каждый запрос к внешнему API
REQUEST_ATTEMPTS = 2  # сколько раз пробуем при сбое сети
MAX_TOPIC_LENGTH = 200

server = FastMCP(name='Topic Experiment')


# ---------------------------------------------------------------- prompts

@server.prompt()
def explain_topic(topic: str) -> str:
    "Генерирует промпт для объяснения темы в дружелюбной манере"
    # FastMCP 3.1 скрывает текст исключений из prompt (клиент видит только
    # "Error rendering prompt"), поэтому при некорректной теме не падаем,
    # а возвращаем промпт, который просит модель уточнить тему у пользователя.
    topic = topic.strip()
    if not topic:
        return ("Пользователь не указал тему для объяснения. Ничего не объясняй: "
                "вежливо попроси его назвать тему.")
    if len(topic) > MAX_TOPIC_LENGTH:
        return (f"Пользователь передал вместо темы текст длиной {len(topic)} символов "
                f"(допускается не более {MAX_TOPIC_LENGTH}). Ничего не объясняй: "
                "вежливо попроси сформулировать тему короче.")
    return (
        f"Объясни тему «{topic}» простым языком для новичка, без технического жаргона.\n"
        "Структура ответа:\n"
        "1. Определение в одном предложении.\n"
        "2. Аналогия из повседневной жизни.\n"
        "3. Два коротких примера.\n"
        "Отвечай на русском языке, не более 150 слов.\n"
        "Если это бессмысленный набор символов, а не тема, — прямо скажи об этом "
        "и попроси уточнить, ничего не выдумывай."
    )


# -------------------------------------------------------------- resources

def read_data_file(filename: str) -> str:
    """Читает файл ресурса и превращает ошибки файловой системы в понятную ошибку MCP."""
    path = DATA_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ResourceError(f"Файл ресурса не найден: {filename}")
    except OSError as error:
        raise ResourceError(f"Не удалось прочитать ресурс {filename}: {error}")


@server.resource("entry-weather-example://weather-letter", mime_type="text/markdown")
def write_3way_intro() -> str:
    """Пример написания приветственного сообщения, если пользователь хочет информацию по погоде"""
    return read_data_file("entry_letter.md")


@server.resource("info-excursion://Paris-trips", mime_type="text/markdown")
def write_excursion_intro() -> str:
    """Информация по экскурсиям в Париже"""
    return read_data_file("paris_trips.md")


@server.resource("info-excursion://Yekaterinburg-trips", mime_type="text/markdown")
def write_yekaterinburg_excursions() -> str:
    """Информация по экскурсиям в Екатеринбурге"""
    return read_data_file("yekaterinburg_trips.md")


# ------------------------------------------------------------------ tools

@server.tool()
async def add(a: int, b: int) -> str:
    """Сумма двух чисел"""
    return str(a + b)


def fetch_json(url: str, params: dict, service: str) -> dict:
    """GET-запрос к внешнему API с таймаутом; любая ошибка превращается в ToolError."""
    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError) as error:
            # сбой сети часто разовый — повторяем запрос, прежде чем сдаться
            if attempt < REQUEST_ATTEMPTS:
                continue
            if isinstance(error, requests.Timeout):
                raise ToolError(f"{service}: нет ответа за {REQUEST_TIMEOUT} с. Попробуйте позже.")
            raise ToolError(f"{service}: нет соединения. Проверьте подключение к интернету.")
        except requests.HTTPError as error:
            raise ToolError(f"{service}: ошибка HTTP {error.response.status_code}.")
        except ValueError:
            raise ToolError(f"{service}: ответ не в формате JSON.")


@server.tool()
def get_weather(city: str) -> str:
    """
    Получает текущую погоду в городе.
    """
    city = city.strip()
    if not city:
        raise ToolError("Название города не указано.")

    # 1 Получаем координаты города
    geo_data = fetch_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": city, "count": 1, "language": "ru"},
        "Геокодирование Open-Meteo",
    )
    results = geo_data.get("results")
    if not results:
        raise ToolError(f"Город «{city}» не найден. Проверьте написание.")

    place = results[0]
    # Показываем, какой именно город нашёл API: по одному названию их может быть несколько
    found = ", ".join(part for part in (place.get("name"), place.get("country")) if part)

    # 2 Получаем погоду
    weather_data = fetch_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current_weather": "true",
        },
        "Погода Open-Meteo",
    )
    current = weather_data.get("current_weather")
    if not current or "temperature" not in current or "windspeed" not in current:
        raise ToolError("Погода Open-Meteo: в ответе нет текущей погоды.")

    return f"Погода в {found}: {current['temperature']}°C, ветер {current['windspeed']} км/ч"


TEMPERATURE_UNITS = {
    "c": "C", "с": "C", "celsius": "C", "цельсий": "C",   # вторая «с» — кириллическая
    "f": "F", "fahrenheit": "F", "фаренгейт": "F",
    "k": "K", "к": "K", "kelvin": "K", "кельвин": "K",
}
UNIT_SIGNS = {"C": "°C", "F": "°F", "K": "K"}


@server.tool()
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """
    Переводит температуру между шкалами Цельсия (C), Фаренгейта (F) и Кельвина (K).
    Пример: value=100, from_unit="C", to_unit="F".
    """
    def normalize(unit: str) -> str:
        key = unit.strip().lower().lstrip("°")
        if key not in TEMPERATURE_UNITS:
            raise ToolError(f"Неизвестная шкала «{unit}». Допустимые значения: C, F, K.")
        return TEMPERATURE_UNITS[key]

    source, target = normalize(from_unit), normalize(to_unit)

    # Всё приводим к Кельвинам: так проще проверить физический смысл значения
    if source == "C":
        kelvin = value + 273.15
    elif source == "F":
        kelvin = (value - 32) * 5 / 9 + 273.15
    else:
        kelvin = value

    if kelvin < -1e-9:
        raise ToolError(
            f"{value:g}{UNIT_SIGNS[source]} ниже абсолютного нуля — такой температуры не бывает."
        )

    if target == "C":
        result = kelvin - 273.15
    elif target == "F":
        result = (kelvin - 273.15) * 9 / 5 + 32
    else:
        result = kelvin

    return f"{value:g}{UNIT_SIGNS[source]} = {round(result, 2):g}{UNIT_SIGNS[target]}"


if __name__ == "__main__":
    if is_stdio == 'True':
        # уровень протокола stdio
        server.run(transport="stdio")
    else:
        # уровень протокола streamable-http
        server.run(transport="streamable-http")
