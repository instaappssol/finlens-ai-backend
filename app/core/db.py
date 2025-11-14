from fastapi import Request
from pymongo.database import Database

def get_db_from_request(request: Request) -> Database:
    return request.app.state.db
