"""Placeholder chat orchestration service."""


class ChatService:
    def __init__(self) -> None:
        self.ready = False

    def prepare_response(self, user_message: str) -> str:
        return (
            "This prototype is not yet connected to the legal-source retrieval pipeline. "
            "It will respond once the knowledge base and chat logic are implemented."
        )
