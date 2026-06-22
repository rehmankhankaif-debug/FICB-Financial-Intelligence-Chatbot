from __future__ import annotations

import pytest

from src.auth.passwords import hash_password, verify_password
from src.auth.service import DuplicateUserError, InvalidCredentialsError, WeakPasswordError, AuthService
from src.storage.sqlite_store import SQLiteStore


def test_password_hashing_uses_non_plaintext_verifiable_hash() -> None:
    stored_hash = hash_password("Secure123!")

    assert "Secure123!" not in stored_hash
    assert verify_password("Secure123!", stored_hash) is True
    assert verify_password("wrong-password", stored_hash) is False


def test_auth_service_registers_and_authenticates_user(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "app.sqlite3")
    service = AuthService(store)

    user = service.register("Kaif@example.com", "Secure123!", "Kaif")
    authenticated = service.authenticate("kaif@example.com", "Secure123!")

    assert user.email == "kaif@example.com"
    assert user.role == "admin"
    assert authenticated.user_id == user.user_id
    assert authenticated.last_login_at is not None
    assert authenticated.password_hash != "Secure123!"


def test_auth_service_rejects_duplicate_email(tmp_path) -> None:
    service = AuthService(SQLiteStore(tmp_path / "app.sqlite3"))

    service.register("kaif@example.com", "Secure123!", "Kaif")

    with pytest.raises(DuplicateUserError):
        service.register("KAIF@example.com", "Secure123!", "Kaif")


def test_auth_service_registers_later_users_as_standard_users(tmp_path) -> None:
    service = AuthService(SQLiteStore(tmp_path / "app.sqlite3"))

    admin = service.register("admin@example.com", "Secure123!", "Admin")
    user = service.register("user@example.com", "Secure123!", "User")

    assert admin.role == "admin"
    assert user.role == "user"


def test_auth_service_rejects_weak_password_and_invalid_login(tmp_path) -> None:
    service = AuthService(SQLiteStore(tmp_path / "app.sqlite3"))

    with pytest.raises(WeakPasswordError):
        service.register("kaif@example.com", "password", "Kaif")

    service.register("kaif@example.com", "Secure123!", "Kaif")

    with pytest.raises(InvalidCredentialsError):
        service.authenticate("kaif@example.com", "wrong-password")
