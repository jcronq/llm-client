from llm_client.agent.memory.sql_backed_memory_objects import SqlMessage


class Interaction:
    def __init__(
        self,
        uid: str,
        user_message: SqlMessage,
        response_message: SqlMessage,
        system_message_ids: list[str],
        relevant_message_ids: list[str],
        recent_message_ids: list[str],
    ):
        self.uid = uid
        self.user_message = user_message
        self.resposne_message = response_message
        self.system_message_ids = system_message_ids
        self.relevant_message_ids = relevant_message_ids
        self.recent_message_ids = recent_message_ids
