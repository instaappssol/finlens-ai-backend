from fastapi import APIRouter, Depends, status
from app.schemas.auth_schema import (
    SignupRequest,
    SignupResponse,
    LoginRequest,
    LoginResponse,
)
from app.services.auth_service import AuthService
from app.core.dependencies import get_auth_service
from app.core.jwt_handler import create_access_token
from app.core.exceptions import (
    BadRequestException,
    InternalServerErrorException,
    ResponseBody,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/signup",
    response_model=ResponseBody,
    status_code=status.HTTP_201_CREATED,
    summary="User Signup",
    description="Register a new user with email, mobile number, and password",
)
def signup(
    signup_data: SignupRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Register a new user.

    Args:
        signup_data: SignupRequest containing email, mobile_number, and password
        auth_service: Injected authentication service

    Returns:
        SignupResponse: Created user information

    Raises:
        HTTPException 400: If email or mobile number already exists
    """
    try:
        user = auth_service.signup(
            name=signup_data.name,
            email=signup_data.email,
            mobile_number=signup_data.mobile_number,
            password=signup_data.password,
        )

        return ResponseBody(
            message="User registered successfully",
            data=SignupResponse(
                id=str(user["_id"]),
                name=user.get("name", ""),
                email=user["email"],
                message="User registered successfully",
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
    summary="User Login",
    description="Authenticate user with email/mobile number and password",
)
def login(
    login_data: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Authenticate user by email or mobile number.

    Args:
        login_data: LoginRequest containing credential (email or mobile) and password
        auth_service: Injected authentication service

    Returns:
        LoginResponse: User information if authentication successful

    Raises:
        HTTPException 401: If credentials are invalid
    """
    try:
        user = auth_service.login(
            credential=login_data.credential, password=login_data.password
        )

        token = create_access_token(user_id=str(user["_id"]), email=user["email"])

        return ResponseBody(
            message="User registered successfully",
            data=LoginResponse(
                id=str(user["_id"]),
                email=user["email"],
                mobile_number=user["mobile_number"],
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
