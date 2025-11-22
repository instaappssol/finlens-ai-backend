from fastapi import APIRouter, Depends, status
from app.schemas.admin_schema import (
    AdminSignupRequest,
    AdminSignupResponse,
    AdminLoginRequest,
    AdminLoginResponse,
)
from app.services.admin_auth_service import AdminAuthService
from app.core.dependencies import get_admin_auth_service
from app.core.jwt_handler import create_access_token
from app.core.exceptions import (
    BadRequestException,
    InternalServerErrorException,
    ResponseBody,
)

router = APIRouter(prefix="/admin/auth", tags=["admin-auth"])


@router.post(
    "/signup",
    response_model=ResponseBody,
    status_code=status.HTTP_201_CREATED,
    summary="Admin Signup",
    description="Register a new admin with name, email, and password",
)
def admin_signup(
    signup_data: AdminSignupRequest,
    auth_service: AdminAuthService = Depends(get_admin_auth_service)
):
    """
    Register a new admin.

    Args:
        signup_data: AdminSignupRequest containing name, email, and password
        auth_service: Injected admin authentication service

    Returns:
        AdminSignupResponse: Created admin information

    Raises:
        HTTPException 400: If email already exists
    """
    try:
        admin = auth_service.signup(
            name=signup_data.name,
            email=signup_data.email,
            password=signup_data.password,
        )

        return ResponseBody(
            message="Admin registered successfully",
            data=AdminSignupResponse(
                id=str(admin["_id"]),
                name=admin["name"],
                email=admin["email"],
                message="Admin registered successfully",
            ),
        )
    except ValueError as e:
        raise BadRequestException(
            message=str(e),
        )
    except Exception as e:
        raise InternalServerErrorException(
            message=str(e),
        )


@router.post(
    "/login",
    response_model=ResponseBody,
    status_code=status.HTTP_200_OK,
    summary="Admin Login",
    description="Authenticate admin with email and password",
)
def admin_login(
    login_data: AdminLoginRequest,
    auth_service: AdminAuthService = Depends(get_admin_auth_service)
):
    """
    Authenticate admin by email.

    Args:
        login_data: AdminLoginRequest containing email and password
        auth_service: Injected admin authentication service

    Returns:
        AdminLoginResponse: Admin information if authentication successful

    Raises:
        HTTPException 401: If credentials are invalid
    """
    try:
        admin = auth_service.login(
            email=login_data.email, password=login_data.password
        )

        token = create_access_token(user_id=str(admin["_id"]), email=admin["email"])

        return ResponseBody(
            message="Admin login successful",
            data=AdminLoginResponse(
                id=str(admin["_id"]),
                name=admin["name"],
                email=admin["email"],
                token=token,
            ),
        )
    except ValueError as e:
        raise BadRequestException(
            message=str(e),
        )
    except Exception as e:
        raise InternalServerErrorException(
            message=str(e),
        )

