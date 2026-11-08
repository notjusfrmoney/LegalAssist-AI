"""Placeholder LLM service.

This project intentionally keeps the service layer simple and explicit so future
retrieval and model calls can be added without extra framework overhead.
"""


class LLMService:
    def __init__(self) -> None:
        self.model_name = "not-configured"

    async def generate(self, *args, **kwargs) -> str:
        return "LLM integration is not configured in this phase."
