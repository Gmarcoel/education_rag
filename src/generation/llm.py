from abc import ABC, abstractmethod
import google.generativeai as genai
from pathlib import Path
import time
import random

from config.settings import Settings


class LanguageModel(ABC):
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class GeminiModel(LanguageModel):
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self._configure_api()
        self.model = genai.GenerativeModel(model_name)
    
    def _configure_api(self):
        genai.configure(api_key=self.api_key)
    
    def generate(self, prompt: str) -> str:
        max_retries = 3
        base_delay = 30  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries - 1:
                        # Extract retry delay from error message if available
                        delay = base_delay + random.uniform(5, 15)  # Add jitter
                        print(f"API quota limit hit. Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                raise e


class LLMService:
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = self._create_model()
    
    def _create_model(self) -> LanguageModel:
        api_key = self._read_api_key()
        return GeminiModel(self.settings.LLM_MODEL, api_key)
    
    def _read_api_key(self) -> str:
        api_key_path = Path(self.settings.API_KEY_FILE)
        if not api_key_path.exists():
            raise FileNotFoundError(f"API key file not found: {api_key_path}")
        
        with open(api_key_path, 'r') as f:
            return f.read().strip()
    
    def generate_response(self, prompt: str) -> str:
        return self.model.generate(prompt)
