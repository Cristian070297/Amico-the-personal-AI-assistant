import os
from dotenv import load_dotenv
from typing import Optional, List

class ConfigLoader:
    """Handles loading configuration from environment variables."""
    def __init__(self, env_file_path=".env"):
        load_dotenv(dotenv_path=env_file_path)
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.fine_tuned_model_id: Optional[str] = os.getenv("fine_tuned_model_id")
        self.embedding_model: str = "text-embedding-ada-002"
        self.gnews_api_key: Optional[str] = os.getenv("GNEWS-API")
        self.fmp_api_key: Optional[str] = os.getenv("FMP-FINANCE-API")
        self.weather_api_key: Optional[str] = os.getenv("weather-api_key")
        self.db_path: str = 'AmIco2.db' # Default, can be overridden if needed

    def validate_keys(self) -> List[str]:
        """Checks for missing essential keys."""
        missing = []
        if not self.openai_api_key: missing.append("OPENAI_API_KEY")
        if not self.fine_tuned_model_id: missing.append("fine_tuned_model_id")
        if not self.gnews_api_key: missing.append("GNEWS-API")
        if not self.fmp_api_key: missing.append("FMP-FINANCE-API")
        if not self.weather_api_key: missing.append("weather-api_key")
        return missing