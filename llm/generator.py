from openai import OpenAI

from config.settings import OPENAI_API_KEY


class LLMGenerator:

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

    def generate(self, prompt):

        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )

        return response.output_text