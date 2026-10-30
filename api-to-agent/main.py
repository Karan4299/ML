import os
from pathlib import Path
from openai import OpenAI
import json
import requests


# 1. Manually find and parse the .env file in your workspace
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip("'\"")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

FREE_MODEL = "openrouter/free"
TOOL_MODEL = "openrouter/free"

def get_weather(city):
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()

        if not geo.get("results"):
            return f"Could not find a city named '{city}'."

        place = geo["results"][0]
        lat, lon = place["latitude"], place["longitude"]

        # Step 2: get the current weather for those coordinates
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            },
            timeout=10,
        ).json()

        now = weather["current"]

        # Step 3: turn the weather code into words
        codes = {
            0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy",
            45: "foggy", 48: "foggy",
            51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
            61: "light rain", 63: "rain", 65: "heavy rain",
            71: "light snow", 73: "snow", 75: "heavy snow",
            80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
            95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
        }
        condition = codes.get(now["weather_code"], "unknown")

        return (
            f"{place['name']}, {place.get('country', '')}: "
            f"{now['temperature_2m']}°C, {condition}, "
            f"humidity {now['relative_humidity_2m']}%, "
            f"wind {now['wind_speed_10m']} km/h"
        )

    except Exception as e:
        return f"Error getting weather: {e}"

HEADERS = {"User-Agent": "MyLearningAgent/1.0 (contact: you@example.com)"}  # Wikipedia requires a User-Agent

def search_wikipedia(query):
    """Search Wikipedia and return a short summary of the best matching article."""
    try:
        # Step 1: search for the best matching article title
        search = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 1,
                "format": "json",
            },
            headers=HEADERS,
            timeout=10,
        ).json()

        results = search.get("query", {}).get("search", [])
        if not results:
            return f"No Wikipedia article found for '{query}'."

        title = results[0]["title"]

        # Step 2: get the summary of that article
        safe_title = requests.utils.quote(title.replace(" ", "_"))
        summary = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}",
            headers=HEADERS,
            timeout=10,
        ).json()

        text = summary.get("extract", "No summary available.")
        url = summary.get("content_urls", {}).get("desktop", {}).get("page", "")

        return f"{title}\n\n{text[:1500]}\n\nSource: {url}"

    except Exception as e:
        return f"Error searching Wikipedia: {e}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current real-time weather (temperature, condition, humidity, wind) for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. Mumbai"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia and get a summary about a person, place, thing, event or topic. Use this for factual or historical information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What is the search for"}
                },
                "required": ["query"],
            },

        }
    }
]

message = 'What is the capital of maharashtra and what is the weather like there now?'

# response = client.chat.completions.create(
#     model=TOOL_MODEL,
#     messages=[{"role": "user", "content": message}],
#     temperature=0.0,
#     tools=tools
# )

# msg = response.choices[0].message
# print(f"Finish reason: {response.choices[0].finish_reason}")
# print(f"Content: {msg.content}")
# print(f"Tool calls: {msg.tool_calls}")
TOOL_FNS = {"get_weather": get_weather, "search_wikipedia": search_wikipedia}
# if msg.tool_calls:
#     tc = msg.tool_calls[0]
#     print(f"\n→ Model wants to call: {tc.function.name}({tc.function.arguments})")
#     print("  It GENERATED JSON requesting a function. Our code must execute it.")

#     fn = TOOL_FNS[tc.function.name]
#     result = fn(**json.loads(tc.function.arguments))
#     conversations = [
#         {"role": "user", "content": message},
#         msg,
#         {"role": "tool", "tool_call_id": tc.id, "content": result}
#     ]

#     r2 = client.chat.completions.create(
#         model=TOOL_MODEL, messages=conversations, tools=tools
#     )

#     print(r2.choices[0].message.content)

#     if r2.choices[0].message.tool_calls:
#         print("Want to call again to", r2.choices[0].message.tool_calls)



conversations = [
    {"role": "user", "content": message}
]

still_running = True

while still_running:
    response = client.chat.completions.create(
        model=TOOL_MODEL,
        messages=conversations,
        temperature=0.0,
        tools=tools
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        tool_asked_to_call = msg.tool_calls[0]
        tool_function = tool_asked_to_call.function.name
        tool_arg = tool_asked_to_call.function.arguments
        conversations.append(msg)

        fn = TOOL_FNS[tool_function]

        if fn:
            result = fn(**json.loads(tool_arg))
            conversations.append({"role": "tool", "tool_call_id": tool_asked_to_call.id, "content": result})
        else:
            still_running = False
    else:
        still_running = False
        print(f"Tool calls: {msg.content}")


