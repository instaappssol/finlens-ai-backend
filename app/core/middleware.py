from typing import List, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.jwt_handler import verify_token
from app.core.exceptions import ResponseBody
from fastapi.responses import JSONResponse
from fastapi import status

exempt_paths = [
    "/auth/login",
    "/auth/signup",
    "/health",
    "/openapi.json",
    "/docs",
]


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to validate JWT token on incoming requests.

    Exempt paths (public endpoints) can be passed via `exempt_paths` kwarg
    when registering the middleware (e.g. in `main.py`).
    """

    def __init__(self, app):
        super().__init__(app)
        self.exempt_paths = set(exempt_paths or [])

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip validation for exempt paths
        path = request.url.path
        if path in self.exempt_paths:
            return await call_next(request)

        # Read Authorization header
        auth_header = request.headers.get("authorization") or request.headers.get(
            "Authorization"
        )
        if not auth_header:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=ResponseBody(
                    message="Authorization header is required",
                    errors=[],
                    data=None,
                ).model_dump(),
            )

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=ResponseBody(
                    message="Invalid authorization header",
                    errors=[],
                    data=None,
                ).model_dump(),
            )

        token = parts[1]
        try:
            payload = verify_token(token)
        except Exception as e:
            # verify_token raises JWT errors for invalid/expired tokens
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=ResponseBody(
                    message=str(e),
                    errors=[],
                    data=None,
                ).model_dump(),
            )

        # Attach user info to request.state for downstream handlers
        request.state.user = payload

        return await call_next(request)
