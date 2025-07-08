from abc import ABC, abstractmethod
import base64
from io import BytesIO
import openai
from fastapi import HTTPException

class AIServiceProvider(ABC):
    @abstractmethod
    async def analyze_image(self, image_base64: str, prompt: str, model: str, output_format: str, max_tokens: int) -> str:
        pass

class OpenAIProvider(AIServiceProvider):
    def __init__(self, api_key: str, base_url: str = None):
        if not api_key:
            raise ValueError("OpenAI API Key is not provided.")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url if base_url else "https://api.openai.com/v1")

    async def analyze_image(self, image_base64: str, prompt: str, model: str, output_format: str, max_tokens: int) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/{output_format};base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except openai.APIError as e:
            raise HTTPException(
                status_code=e.status_code if hasattr(e, 'status_code') else 500,
                detail=f"OpenAI API Error: {e.response.json() if hasattr(e, 'response') and hasattr(e.response, 'json') else str(e)}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI analysis failed: {str(e)}")
