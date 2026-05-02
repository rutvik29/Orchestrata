"""
Weather tool using the free wttr.in API (no API key required).

Fetches current weather conditions and a 3-day forecast for any city.
"""

import logging
import urllib.parse
import urllib.request
import json
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

WTTR_BASE_URL = "https://wttr.in"


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather and 3-day forecast for a city.

    Uses the free wttr.in API -- no API key required.

    Args:
        city: City name (e.g., 'London', 'New York', 'Tokyo').
              Can also accept coordinates like '48.8566,2.3522'.

    Returns:
        A formatted string with current conditions and forecast,
        or an error message if the city is not found.
    """
    logger.info(f"[get_weather] Fetching weather for: {city}")

    try:
        encoded_city = urllib.parse.quote(city)
        url = f"{WTTR_BASE_URL}/{encoded_city}?format=j1"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Orchestrata-Agent/1.0"},
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        # Parse current condition
        current = data["current_condition"][0]
        weather_desc = current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]
        temp_f = current["temp_F"]
        feels_like_c = current["FeelsLikeC"]
        humidity = current["humidity"]
        wind_speed_kmph = current["windspeedKmph"]
        wind_dir = current["winddir16Point"]
        visibility = current["visibility"]
        uv_index = current["uvIndex"]

        # Parse location
        nearest_area = data.get("nearest_area", [{}])[0]
        area_name = nearest_area.get("areaName", [{}])[0].get("value", city)
        country = nearest_area.get("country", [{}])[0].get("value", "")

        # Parse 3-day forecast
        forecast_lines = []
        for day_data in data.get("weather", [])[:3]:
            date = day_data["date"]
            max_c = day_data["maxtempC"]
            min_c = day_data["mintempC"]
            desc = day_data["hourly"][4]["weatherDesc"][0]["value"]  # noon forecast
            forecast_lines.append(f"  {date}: {desc}, {min_c}C - {max_c}C")

        forecast_text = "\n".join(forecast_lines) if forecast_lines else "  Forecast unavailable"

        return (
            f"Weather for {area_name}, {country}:\n"
            f"  Condition:   {weather_desc}\n"
            f"  Temperature: {temp_c}C ({temp_f}F), feels like {feels_like_c}C\n"
            f"  Humidity:    {humidity}%\n"
            f"  Wind:        {wind_speed_kmph} km/h {wind_dir}\n"
            f"  Visibility:  {visibility} km\n"
            f"  UV Index:    {uv_index}\n"
            f"\n3-Day Forecast:\n{forecast_text}"
        )

    except urllib.error.HTTPError as e:
        logger.warning(f"[get_weather] HTTP error {e.code} for city '{city}'")
        return f"Error: Could not fetch weather for '{city}' (HTTP {e.code}). Check the city name."
    except urllib.error.URLError as e:
        logger.error(f"[get_weather] URL error: {e}")
        return f"Error: Network error fetching weather data -- {e.reason}"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"[get_weather] Parse error: {e}")
        return f"Error: Could not parse weather response for '{city}'."
    except Exception as e:
        logger.exception(f"[get_weather] Unexpected error")
        return f"Error fetching weather for '{city}': {e}"
