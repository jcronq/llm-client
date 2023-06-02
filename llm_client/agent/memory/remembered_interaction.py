from datetime import datetime

from pydantic import BaseModel


class RememberedInteraction(BaseModel):
    uid: str
    created_at: datetime
    user_message: str
    response_message: str
