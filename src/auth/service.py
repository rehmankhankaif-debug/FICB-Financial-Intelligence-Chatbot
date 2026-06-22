from __future__ import annotations

import re
import sqlite3
from typing import Optional

from src.auth.passwords import hash_password, validate_password_strength, verify_password
from src.models.user import User
from src.storage.sqlite_store import SQLiteStore


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    pass


class DuplicateUserError(AuthError):
    pass


class InvalidCredentialsError(AuthError):
    pass


class WeakPasswordError(AuthError):
    pass


class AuthService:
    def __init__(self, store: Optional[SQLiteStore] = None) -> None:
        self.store = store or SQLiteStore()

    def register(self, email: str, password: str, display_name: str = "") -> User:
        normalized_email = self._normalize_email(email)
        if not EMAIL_RE.fullmatch(normalized_email):
            raise AuthError("Enter a valid email address.")
        try:
            validate_password_strength(password)
        except ValueError as exc:
            raise WeakPasswordError(str(exc)) from exc

        name = str(display_name or "").strip() or normalized_email.split("@", 1)[0]
        role = "admin" if self.store.count_users() == 0 else "user"
        try:
            user = self.store.create_user(
                email=normalized_email,
                display_name=name,
                password_hash=hash_password(password),
                role=role,
            )
        except sqlite3.IntegrityError as exc:
            raise DuplicateUserError("An account with this email already exists.") from exc

        self.store.record_audit_event("auth.registered", {"email": normalized_email, "role": user.role}, user_id=user.user_id)
        return user

    def authenticate(self, email: str, password: str) -> User:
        normalized_email = self._normalize_email(email)
        user = self.store.get_user_by_email(normalized_email)
        if user is None or not user.is_active or not verify_password(password, user.password_hash):
            self.store.record_audit_event("auth.login_failed", {"email": normalized_email})
            raise InvalidCredentialsError("Invalid email or password.")
        self.store.update_last_login(user.user_id)
        refreshed = self.store.get_user_by_id(user.user_id) or user
        self.store.record_audit_event("auth.login_succeeded", {"email": normalized_email}, user_id=user.user_id)
        return refreshed

    def get_user(self, user_id: str) -> Optional[User]:
        return self.store.get_user_by_id(user_id)

    def _normalize_email(self, email: str) -> str:
        return str(email or "").strip().lower()
