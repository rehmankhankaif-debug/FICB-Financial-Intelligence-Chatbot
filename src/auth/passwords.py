from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Tuple


HASH_SCHEME = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 240_000
SALT_BYTES = 16
MIN_PASSWORD_LENGTH = 8


def hash_password(password: str, iterations: int = DEFAULT_ITERATIONS) -> str:
    if not password:
        raise ValueError("Password cannot be empty.")
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    salt_value = base64.b64encode(salt).decode("ascii")
    hash_value = base64.b64encode(digest).decode("ascii")
    return "{0}${1}${2}${3}".format(HASH_SCHEME, iterations, salt_value, hash_value)


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iterations, salt, expected_hash = _parse_hash(stored_hash)
        if scheme != HASH_SCHEME:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            str(password or "").encode("utf-8"),
            base64.b64decode(salt.encode("ascii")),
            iterations,
        )
        calculated = base64.b64encode(digest).decode("ascii")
        return hmac.compare_digest(calculated, expected_hash)
    except Exception:
        return False


def validate_password_strength(password: str) -> None:
    value = str(password or "")
    if len(value) < MIN_PASSWORD_LENGTH:
        raise ValueError("Password must be at least {0} characters long.".format(MIN_PASSWORD_LENGTH))
    if value.isdigit() or value.isalpha():
        raise ValueError("Password must include a mix of letters, numbers, or symbols.")


def _parse_hash(stored_hash: str) -> Tuple[str, int, str, str]:
    scheme, iterations, salt, expected_hash = str(stored_hash or "").split("$", 3)
    return scheme, int(iterations), salt, expected_hash
