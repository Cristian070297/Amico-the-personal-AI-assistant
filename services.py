# services.py
import openai
import requests
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any, Optional, Tuple

# --- OpenAI Service (Embeddings & Chat) ---

class OpenAIService:
    """Handles interactions with OpenAI API for embeddings and chat."""
    def __init__(self, api_key: Optional[str], embedding_model: str, fine_tuned_model_id: Optional[str]):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        openai.api_key = api_key
        self.embedding_model = embedding_model
        self.fine_tuned_model_id = fine_tuned_model_id
        self.embedding_dim = 1536 # Specific to ada-002
        self.default_embedding = [0.0] * self.embedding_dim

    def get_embedding(self, text: str) -> List[float]:
        """Generates embeddings using OpenAI API."""
        if not text or not isinstance(text, str):
            print("Warning: Invalid text received for embedding. Returning zero vector.")
            return self.default_embedding[:] # Return copy

        text = text.replace("\n", " ")
        try:
            response = openai.Embedding.create(input=[text], model=self.embedding_model)
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error generating embedding for text '{text[:50]}...': {e}")
            return self.default_embedding[:] # Return copy

    def get_chat_completion(self, messages: List[Dict[str, str]],
                            model_override: Optional[str] = None,
                            max_tokens: int = 150, temperature: float = 0.7) -> str: # Changed return type hint
        """Gets chat completion using the fine-tuned model (or override)."""
        model_to_use = model_override if model_override else self.fine_tuned_model_id
        if not model_to_use:
            msg = "Error: No model specified for chat completion (fine-tuned ID missing and no override)."
            print(msg)
            return msg # Return error message

        try:
            response = openai.ChatCompletion.create(
                model=model_to_use,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            response_text = response['choices'][0]['message']['content'].strip()
            return response_text
        except openai.error.OpenAIError as e:
            err_msg = f"Sorry, I encountered an API error trying to process that: {e}"
            print(f"OpenAI API error (using model {model_to_use}): {e}")
            return err_msg # Return error message
        except Exception as e:
            err_msg = "Sorry, an unexpected error occurred while generating the response."
            print(f"Unexpected error during OpenAI request (model {model_to_use}): {e}")
            return err_msg # Return error message


# --- External API Service ---

class ExternalApiService:
    """Handles requests to external APIs (Weather, News, Stocks) with caching."""
    def __init__(self, gnews_key: Optional[str], fmp_key: Optional[str], weather_key: Optional[str]):
        self.gnews_key = gnews_key
        self.fmp_key = fmp_key
        self.weather_key = weather_key
        self.cache: Dict[str, Dict[str, Tuple[datetime, Any]]] = {
            "weather": {}, "news": {}, "stocks": {}
        }
        self.cache_expiry = {
            "weather": timedelta(hours=1),
            "news": timedelta(hours=1),
            "stocks": timedelta(minutes=15)
        }

    def _make_request(self, url: str, params: Optional[Dict] = None,
                      service_name: Optional[str] = None, cache_key: Optional[str] = None,
                      expiry_delta: Optional[timedelta] = None, max_retries: int = 3) -> Optional[Any]:
        """Makes an API request with caching and retry logic."""
        # Check cache
        if service_name and cache_key and service_name in self.cache and cache_key in self.cache[service_name]:
            timestamp, data = self.cache[service_name][cache_key]
            effective_expiry = expiry_delta or self.cache_expiry.get(service_name, timedelta(minutes=5))
            if datetime.now() - timestamp < effective_expiry:
                print(f"Cache hit for {service_name}: {cache_key}")
                return data

        # Make API call
        print(f"Cache miss or expired for {service_name}: {cache_key}. Making API call...")
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()

                # Update cache
                if service_name and cache_key:
                    self.cache[service_name][cache_key] = (datetime.now(), data)
                return data
            except requests.exceptions.Timeout:
                print(f"Attempt {attempt + 1}/{max_retries}: Request timed out.")
            except requests.exceptions.HTTPError as e:
                print(f"Attempt {attempt + 1}/{max_retries}: HTTP Error: {e.response.status_code} {e.response.reason}")
                if 400 <= e.response.status_code < 500:
                    print("Client error, not retrying.")
                    try: return e.response.json()
                    except ValueError: return {"error": f"HTTP {e.response.status_code}: {e.response.reason}"}
                # Removed break for server errors, allow retry
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1}/{max_retries}: Request failed: {e}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. API call failed.")
                return None # Indicate failure

    def get_stock_data(self, symbol: str) -> str:
        """Fetches stock quote data using FMP API with caching."""
        if not self.fmp_key: return "Financial Modeling Prep API key is missing."

        symbol = symbol.upper()
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
        params = {"apikey": self.fmp_key}
        data_list = self._make_request(url, params=params, service_name="stocks", cache_key=symbol+"_fmp_quote")

        if data_list and isinstance(data_list, list) and len(data_list) > 0:
            try:
                quote = data_list[0]
                timestamp = quote.get("timestamp")
                trade_time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else "N/A"
                return (
                    f"Stock Symbol: {symbol} (via FMP).\n"
                    f"Current Price: ${quote.get('price', 'N/A')} (as of {trade_time_str}).\n"
                    f"Open: ${quote.get('open', 'N/A')}, High: ${quote.get('dayHigh', 'N/A')}, Low: ${quote.get('dayLow', 'N/A')}.\n"
                    f"(Previous Close: ${quote.get('previousClose', 'N/A')})"
                )
            except (KeyError, IndexError, TypeError) as e:
                print(f"Error parsing FMP stock data for {symbol}: {e}")
                return f"Could not parse stock quote from FMP for {symbol}."
        elif isinstance(data_list, list) and len(data_list) == 0:
             return f"Could not find stock data from FMP for symbol: {symbol}. Is the symbol correct?"
        else:
            error_msg = "Unknown error"
            if isinstance(data_list, dict): error_msg = data_list.get("Error Message", str(data_list))
            elif data_list is None: error_msg = "API request failed after retries."
            else: error_msg = "Invalid response format"
            print(f"Failed FMP stock fetch for {symbol}. Response: {error_msg}")
            return f"Failed to fetch stock data from FMP for {symbol}. Error: {error_msg}"

    def get_general_news(self) -> List[Dict[str, Any]]:
        """Fetches top headlines from GNews API with caching."""
        if not self.gnews_key:
            print("GNews API key is missing.")
            return []

        url = "https://gnews.io/api/v4/top-headlines"
        params = {"lang": "en", "country": "gb", "max": 10, "apikey": self.gnews_key}
        data = self._make_request(url, params=params, service_name="news", cache_key="gnews_top_gb_headlines")

        if data and "articles" in data:
            print(f"DEBUG: GNews fetched {len(data['articles'])} articles.")
            # Ensure articles is a list before returning
            articles = data.get("articles")
            return articles if isinstance(articles, list) else []
        else:
            error_msg = data.get("errors", "Unknown error") if isinstance(data, dict) else "No response/invalid format"
            print(f"Failed to fetch news from GNews. Response: {error_msg}")
            return []

    def get_weather(self, location: str) -> str:
        """Fetches weather data from WeatherAPI.com API with caching."""
        if not self.weather_key: return "WeatherAPI.com API key is missing."

        url = "http://api.weatherapi.com/v1/current.json"
        params = {"key": self.weather_key, "q": location, "aqi": "no"}
        data = self._make_request(url, params=params, service_name="weather", cache_key=location+"_weatherapi")

        if data and "current" in data and "location" in data:
            try:
                curr = data["current"]
                loc = data["location"]
                return (
                    f"Current weather in {loc.get('name', 'N/A')}, {loc.get('country', 'N/A')} (as of {curr.get('last_updated', 'N/A')}):\n"
                    f"It's currently {curr.get('condition', {}).get('text', 'N/A')} with a temperature of {curr.get('temp_c', 'N/A')}°C, "
                    f"but feels like {curr.get('feelslike_c', 'N/A')}°C.\n"
                    f"Humidity is {curr.get('humidity', 'N/A')}% and the wind speed is {curr.get('wind_kph', 'N/A')} km/h."
                )
            except KeyError as e:
                print(f"Error parsing WeatherAPI.com data for {location}: Missing key {e}")
                return f"Could not parse weather details from WeatherAPI.com for {location}."
        elif data and "error" in data:
            error_info = data.get("error", {}).get("message", "Unknown API error")
            return f"Could not retrieve weather from WeatherAPI.com for {location}. API Error: {error_info}"
        else:
            error_msg = "Unknown error or invalid format"
            if data is None: error_msg = "API request failed after retries."
            print(f"DEBUG: Unexpected weather response: {data}")
            return f"Failed to fetch weather data from WeatherAPI.com for {location}. Error: {error_msg}"