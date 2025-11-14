from fastapi import APIRouter, Request, HTTPException, status
from app.schemas.auth_schema import SignupRequest, SignupResponse, LoginRequest, LoginResponse
from app.services.auth_service import AuthService
from app.core.db import get_db_from_request

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/signup",
    response_model=SignupResponse,
    status_code=status.HTTP_201_CREATED,
    summary="User Signup",
    description="Register a new user with email, mobile number, and password"
)
def signup(request: Request, signup_data: SignupRequest):
    """
    Register a new user.
    
    Args:
        signup_data: SignupRequest containing email, mobile_number, and password
        
    Returns:
        SignupResponse: Created user information
        
    Raises:
        HTTPException 400: If email or mobile number already exists
    """
    try:
        db = get_db_from_request(request)
        auth_service = AuthService(db)
        
        user = auth_service.signup(
            email=signup_data.email,
            mobile_number=signup_data.mobile_number,
            password=signup_data.password
        )
        
        return SignupResponse(
            id=str(user['_id']),
            email=user['email'],
            message="User registered successfully"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during signup: {str(e)}"
        )


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="User Login",
    description="Authenticate user with email/mobile number and password"
)
def login(request: Request, login_data: LoginRequest):
    """
    Authenticate user by email or mobile number.
    
    Args:
        login_data: LoginRequest containing credential (email or mobile) and password
        
    Returns:
        LoginResponse: User information if authentication successful
        
    Raises:
        HTTPException 401: If credentials are invalid
    """
    try:
        db = get_db_from_request(request)
        auth_service = AuthService(db)
        
        user = auth_service.login(
            credential=login_data.credential,
            password=login_data.password
        )
        
        return LoginResponse(
            id=str(user['_id']),
            email=user['email'],
            mobile_number=user['mobile_number'],
            message="Login successful"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during login: {str(e)}"
        )
