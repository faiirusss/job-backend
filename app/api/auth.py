from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import UserAccount
from app.schemas import AuthResponse, LoginRequest, RegisterRequest, UserDTO
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_dto(user: UserAccount) -> UserDTO:
    return UserDTO(id=user.id, email=user.email, name=user.name)


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        auth_service.COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=auth_service.SESSION_DAYS * 24 * 60 * 60,
        path="/",
    )


@router.post("/register", response_model=AuthResponse, status_code=201)
async def register(
    req: RegisterRequest,
    response: Response,
    session: AsyncSession = Depends(get_db),
) -> AuthResponse:
    try:
        user = await auth_service.register_user(
            session, email=req.email, password=req.password, name=req.name
        )
    except auth_service.EmailAlreadyExistsError as e:
        raise HTTPException(
            status_code=409,
            detail={"error": {"code": "EMAIL_EXISTS", "message": "Email already registered"}},
        ) from e
    token = await auth_service.create_session(session, user.id)
    _set_session_cookie(response, token)
    return AuthResponse(user=_user_dto(user))


@router.post("/login", response_model=AuthResponse)
async def login(
    req: LoginRequest,
    response: Response,
    session: AsyncSession = Depends(get_db),
) -> AuthResponse:
    try:
        user = await auth_service.authenticate_user(session, email=req.email, password=req.password)
    except auth_service.InvalidCredentialsError as e:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "INVALID_CREDENTIALS", "message": "Invalid email or password"}},
        ) from e
    token = await auth_service.create_session(session, user.id)
    _set_session_cookie(response, token)
    return AuthResponse(user=_user_dto(user))


@router.post("/logout", status_code=204)
async def logout(
    response: Response,
    session: AsyncSession = Depends(get_db),
    token: str | None = Cookie(default=None, alias=auth_service.COOKIE_NAME),
) -> Response:
    await auth_service.revoke_session(session, token)
    response.delete_cookie(auth_service.COOKIE_NAME, path="/")
    response.status_code = 204
    return response


@router.get("/me", response_model=UserDTO)
async def me(current_user: UserAccount = Depends(get_current_user)) -> UserDTO:
    return _user_dto(current_user)
