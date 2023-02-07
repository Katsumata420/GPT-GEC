from pydantic import BaseModel

class OpenAIConfig(BaseModel):
    api_key: str
    max_token: int
    model: str = "text-davinci-003"
    temperature: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
