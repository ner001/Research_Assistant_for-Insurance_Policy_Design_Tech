import os
import json
import requests

class LLMHandler:
    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url

    def call_llm(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(self.api_url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

def create_llm_handler() -> LLMHandler:
    api_key = os.getenv("LLM_API_KEY")
    api_url = os.getenv("LLM_API_URL")
    return LLMHandler(api_key, api_url)