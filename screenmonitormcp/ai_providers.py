from abc import ABC, abstractmethod
import base64
from io import BytesIO
import openai
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class AIServiceProvider(ABC):
    @abstractmethod
    async def analyze_image(self, image_base64: str, prompt: str, model: str = None, output_format: str = "png", max_tokens: int = 1000, additional_images: list = None) -> str:
        pass

class OpenAIProvider(AIServiceProvider):
    def __init__(self, api_key: str, base_url: str = None):
        if not api_key:
            raise ValueError("OpenAI API Key is not provided.")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url if base_url else "https://api.openai.com/v1")

    async def analyze_image(self, image_base64: str, prompt: str, model: str = None, output_format: str = "png", max_tokens: int = 1000, additional_images: list = None) -> str:
        try:
            # Create content array with text and primary image
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{output_format};base64,{image_base64}"},
                },
            ]

            # Add additional images if provided
            if additional_images and len(additional_images) > 0:
                for img_base64 in additional_images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{output_format};base64,{img_base64}"},
                    })

            logger.info(f"Making API call with model: {model}, max_tokens: {max_tokens}")

            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                max_tokens=max_tokens,
            )

            # Debug logging
            logger.info(f"API response received: {type(response)}")
            logger.info(f"Response attributes: {dir(response) if response else 'None'}")

            # Enhanced error checking for response
            if not response:
                raise ValueError("API response is None")

            if not hasattr(response, 'choices'):
                logger.error(f"Response object has no 'choices' attribute. Available attributes: {dir(response)}")
                raise ValueError("API response has no choices attribute")

            if not response.choices:
                logger.error(f"Response choices is None or empty: {response.choices}")
                raise ValueError("API response has no choices")

            if len(response.choices) == 0:
                raise ValueError("API response choices list is empty")

            choice = response.choices[0]
            if not hasattr(choice, 'message') or not choice.message:
                raise ValueError("API response choice has no message")

            if not hasattr(choice.message, 'content') or choice.message.content is None:
                raise ValueError("API response message has no content")

            return choice.message.content

        except openai.APIError as e:
            error_detail = f"OpenAI API Error: {str(e)}"
            if hasattr(e, 'response') and e.response:
                try:
                    error_detail = f"OpenAI API Error: {e.response.json()}"
                except:
                    error_detail = f"OpenAI API Error: {str(e)}"
            raise HTTPException(
                status_code=e.status_code if hasattr(e, 'status_code') else 500,
                detail=error_detail
            )
        except ValueError as e:
            # Handle our custom validation errors
            raise HTTPException(status_code=500, detail=f"OpenAI response validation failed: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI analysis failed: {str(e)}")
