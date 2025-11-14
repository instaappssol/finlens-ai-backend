"""
Global exception handlers for FastAPI
"""
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from app.core.exceptions import UnprocessableEntityException, BadRequestException, InternalServerErrorException, ResponseBody


def register_exception_handlers(app: FastAPI):
    """Register all exception handlers to the FastAPI app"""

    @app.exception_handler(UnprocessableEntityException)
    async def unprocessable_entity_exception_handler(request, exc: UnprocessableEntityException):
        """Handle UnprocessableEntityException (422)"""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ResponseBody(
                message=exc.message,
                errors=exc.errors,
                data=None
            ).model_dump()
        )

    @app.exception_handler(BadRequestException)
    async def bad_request_exception_handler(request, exc: BadRequestException):
        """Handle BadRequestException (400)"""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ResponseBody(
                message=exc.message,
                errors=exc.errors,
                data=None
            ).model_dump()
        )

    @app.exception_handler(InternalServerErrorException)
    async def internal_server_error_exception_handler(request, exc: InternalServerErrorException):
        """Handle InternalServerErrorException (500)"""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBody(
                message=exc.message,
                errors=exc.errors,
                data=None
            ).model_dump()
        )
