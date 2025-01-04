import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIClientSingleton:
    """
    Simple singleton wrapper around the OpenAI client.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("Creating a new OpenAI client instance...")
            cls._instance = super(OpenAIClientSingleton, cls).__new__(cls)
            # Initialize the OpenAI client only once
            cls._instance._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return cls._instance

    def __getattr__(self, name):
        """
        This allows your singleton class to proxy all calls/attributes
        to the underlying OpenAI client seamlessly. E.g.:

            OpenAIClientSingleton().files.create(...)
        """
        return getattr(self._client, name)


client = OpenAIClientSingleton()
