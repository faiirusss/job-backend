from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuthSession, CV, CoverLetter, MatchResult, SearchQuery, UserAccount

COOKIE_NAME = "jhai_session"
SESSION_DAYS = 30


class EmailAlreadyExistsError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


class InvalidSessionError(Exception):
    pass


try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError

    _password_hasher = PasswordHasher()
except Exception:  # pragma: no cover - exercised only when optional dep is absent
    PasswordHasher = None  # type: ignore[assignment]
    VerifyMismatchError = ValueError  # type: ignore[assignment]
    _password_hasher = None


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    if _password_hasher is not None:
        return _password_hasher.hash(password)
    salt = secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=salt.encode(), n=2**14, r=8, p=1)
    return f"scrypt${salt}${digest.hex()}"


def verify_password(password_hash: str, password: str) -> bool:
    if password_hash.startswith("$argon2") and _password_hasher is not None:
        try:
            return bool(_password_hasher.verify(password_hash, password))
        except VerifyMismatchError:
            return False
    if password_hash.startswith("scrypt$"):
        try:
            _, salt, expected = password_hash.split("$", 2)
            digest = hashlib.scrypt(password.encode(), salt=salt.encode(), n=2**14, r=8, p=1)
            return hmac.compare_digest(digest.hex(), expected)
        except Exception:
            return False
    return False


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def register_user(
    session: AsyncSession, *, email: str, password: str, name: str | None = None
) -> UserAccount:
    email = normalize_email(email)
    existing = (
        await session.execute(select(UserAccount).where(UserAccount.email == email))
    ).scalar_one_or_none()
    if existing is not None:
        raise EmailAlreadyExistsError(email)

    user_count = await session.scalar(select(func.count(UserAccount.id)))
    user = UserAccount(email=email, name=(name or "").strip() or None, password_hash=hash_password(password))
    session.add(user)
    await session.flush()
    await session.refresh(user)

    if int(user_count or 0) == 0:
        await claim_legacy_rows(session, user.id)

    return user


async def authenticate_user(session: AsyncSession, *, email: str, password: str) -> UserAccount:
    user = (
        await session.execute(select(UserAccount).where(UserAccount.email == normalize_email(email)))
    ).scalar_one_or_none()
    if user is None or not verify_password(user.password_hash, password):
        raise InvalidCredentialsError()
    return user


async def create_session(session: AsyncSession, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    row = AuthSession(
        user_id=user_id,
        token_hash=hash_session_token(token),
        expires_at=datetime.now(UTC) + timedelta(days=SESSION_DAYS),
    )
    session.add(row)
    await session.flush()
    return token


async def get_user_by_session_token(session: AsyncSession, token: str | None) -> UserAccount:
    if not token:
        raise InvalidSessionError()
    now = datetime.now(UTC)
    row = (
        await session.execute(
            select(AuthSession, UserAccount)
            .join(UserAccount, UserAccount.id == AuthSession.user_id)
            .where(
                AuthSession.token_hash == hash_session_token(token),
                AuthSession.revoked_at.is_(None),
                AuthSession.expires_at > now,
            )
        )
    ).first()
    if row is None:
        raise InvalidSessionError()
    return row[1]


async def revoke_session(session: AsyncSession, token: str | None) -> None:
    if not token:
        return
    await session.execute(
        update(AuthSession)
        .where(AuthSession.token_hash == hash_session_token(token), AuthSession.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )


async def claim_legacy_rows(session: AsyncSession, user_id: int) -> None:
    for model in (CV, SearchQuery, MatchResult, CoverLetter):
        await session.execute(
            update(model).where(model.user_id.is_(None)).values(user_id=user_id)
        )
