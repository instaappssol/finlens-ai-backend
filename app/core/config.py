from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    APP_NAME: str = "Finlen's API"
    APP_VERSION: str = "0.1.0"
    MONGO_URI: str = Field(..., env='MONGO_URI')
    MONGO_DB: str = Field(..., env='MONGO_DB')

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
