"""Weather tool using Open-Meteo (no API key required)."""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

from assistant.logging import get_logger

log = get_logger(__name__)

WMO_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with light hail",
    99: "thunderstorm with heavy hail",
}


def _describe_code(code: int | None) -> str:
    return WMO_CODES.get(code or 0, "unknown conditions")


@tool
def get_weather(city: str, days: int = 1) -> str:
    """Get current conditions and a daily forecast for a city by name."""
    try:
        geo = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=15.0,
        )
        geo.raise_for_status()
        results = geo.json().get("results", [])
        if not results:
            return f"I couldn't find a city named {city}."
        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        name = place.get("name", city)
        country = place.get("country", "")
        forecast_days = max(1, min(int(days), 7))

        forecast = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m"
                ),
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                "forecast_days": forecast_days,
                "timezone": "auto",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
            },
            timeout=15.0,
        )
        forecast.raise_for_status()
        data = forecast.json()
        current = data.get("current", {})
        temp = current.get("temperature_2m")
        feels = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        condition = _describe_code(current.get("weather_code"))

        parts = [
            f"In {name}{', ' + country if country else ''}, it is currently {condition} "
            f"with a temperature of {temp} degrees Fahrenheit"
        ]
        if feels is not None:
            parts.append(f"feels like {feels}")
        if humidity is not None:
            parts.append(f"humidity {humidity} percent")
        if wind is not None:
            parts.append(f"wind {wind} miles per hour")
        summary = ", ".join(parts) + "."

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        codes = daily.get("weather_code", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        if dates:
            summary += " Forecast:"
            for index, date in enumerate(dates):
                label = "Today" if index == 0 else date
                high = highs[index] if index < len(highs) else "?"
                low = lows[index] if index < len(lows) else "?"
                code = _describe_code(codes[index] if index < len(codes) else None)
                summary += f" {label}: {code}, high {high}, low {low}."
        return summary
    except httpx.HTTPError as exc:
        log.warning("Weather request failed: %s", exc)
        return "I couldn't reach the weather service right now."
    except Exception as exc:
        log.exception("Weather tool error: %s", exc)
        return "Something went wrong while fetching the weather."
