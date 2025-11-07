import os

class Settings:
    def __init__(self):
        self.api_key = os.getenv("API_KEY", "AIzaSyBh1P1HxCNiC1ZF5K5Jt2me60LFOKAMOpw")
        self.api_url = os.getenv("API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent")
        self.model_name = os.getenv("MODEL_NAME", "llama3")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))

settings = Settings()