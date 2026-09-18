import requests

from src.config import WEATHER_API_TIMEOUT


def get_weather(latitude: float, longitude: float) -> dict:
    """Retrieve current weather information from Open-Meteo."""

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&current=temperature_2m,"
        "relative_humidity_2m,"
        "wind_speed_10m,"
        "precipitation"
    )

    response = requests.get(url, timeout = WEATHER_API_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    current = payload.get("current", {})

    return {
        "latitude": latitude,
        "longitude": longitude,
        "temperature_c": current.get("temperature_2m"),
        "humidity_percent": current.get("relative_humidity_2m"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "precipitation_mm": current.get("precipitation")
    }


def format_weather(weather: dict) -> str:
    return (
        "CURRENT WEATHER\n"
        f"Location: "
        f"{weather['latitude']}, "
        f"{weather['longitude']}\n"
        f"Temperature: "
        f"{weather['temperature_c']} °C\n"
        f"Humidity: "
        f"{weather['humidity_percent']}%\n"
        f"Wind: "
        f"{weather['wind_speed_kmh']} km/h\n"
        f"Precipitation: "
        f"{weather['precipitation_mm']} mm"
    )


def get_weather_text(latitude: float, longitude: float) -> str:
    try:
        weather = get_weather(latitude, longitude)
        return format_weather(weather)

    except Exception as error:
        return (
            "Weather API unavailable.\n"
            f"Error: {error}"
        )


if __name__ == "__main__":
    print(get_weather_text(33.77, -118.19))