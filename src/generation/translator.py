from abc import ABC, abstractmethod
import google.generativeai as genai
from pathlib import Path

from config.settings import Settings


class Translator(ABC):
    
    @abstractmethod
    def translate_to_english(self, text: str) -> str:
        pass
    
    @abstractmethod
    def translate_to_spanish(self, text: str) -> str:
        pass


class GeminiTranslator(Translator):
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._configure_api()
        self.model = genai.GenerativeModel("gemini-1.5-flash")
    
    def _configure_api(self):
        api_key = self._read_api_key()
        genai.configure(api_key=api_key)
    
    def _read_api_key(self) -> str:
        api_key_path = Path(self.settings.API_KEY_FILE)
        if not api_key_path.exists():
            raise FileNotFoundError(f"API key file not found: {api_key_path}")
        
        with open(api_key_path, 'r') as f:
            return f.read().strip()
    
    def translate_to_english(self, text: str) -> str:
        prompt = f"""Translate the following text to English. If the text is already in English, return it unchanged. Only return the translation, no explanations:

Text: {text}"""
        
        response = self.model.generate_content(prompt)
        return response.text.strip()
    
    def translate_to_spanish(self, text: str) -> str:
        prompt = f"""Translate the following text to Spanish. Maintain technical accuracy and educational tone. Only return the translation, no explanations:

Text: {text}"""
        
        response = self.model.generate_content(prompt)
        return response.text.strip()


class TranslationService:
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.translator = self._create_translator()
    
    def _create_translator(self) -> Translator:
        return GeminiTranslator(self.settings)
    
    def process_multilingual_query(self, query: str) -> str:
        return self.translator.translate_to_english(query)
    
    def translate_response_to_spanish(self, response: str) -> str:
        return self.translator.translate_to_spanish(response)
