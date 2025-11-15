from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    APP_NAME: str = "Finlen's API"
    APP_VERSION: str = "0.1.0"
    MONGO_URI: str = Field(..., env='MONGO_URI')
    MONGO_DB: str = Field(..., env='MONGO_DB')
    JWT_SECRET_KEY: str = Field(..., env='JWT_SECRET_KEY')
    JWT_ALGORITHM: str = Field(default="HS256", env='JWT_ALGORITHM')
    JWT_EXPIRATION_HOURS: int = Field(default=24, env='JWT_EXPIRATION_HOURS')
    MODELS_DIR: str = Field(default="models", env='MODELS_DIR')

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
