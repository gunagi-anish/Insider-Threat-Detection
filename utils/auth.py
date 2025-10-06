import hashlib
import hmac
import os
from typing import Dict, Optional


class InMemoryAuthStore:
    """Simple in-memory user store with salted password hashing.

    Note: This store is process-local and resets on app restart.
    """

    def __init__(self) -> None:
        self._users: Dict[str, Dict[str, str]] = {}

    def _hash_password(self, password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100_000).hex()

    def _generate_salt(self) -> str:
        return os.urandom(16).hex()

    def create_user(self, username: str, password: str) -> Optional[str]:
        username = username.strip().lower()
        if not username or not password:
            return "Username and password are required"
        if username in self._users:
            return "User already exists"
        salt = self._generate_salt()
        password_hash = self._hash_password(password, salt)
        self._users[username] = {"salt": salt, "password_hash": password_hash}
        return None

    def verify_user(self, username: str, password: str) -> bool:
        username = username.strip().lower()
        user = self._users.get(username)
        if not user:
            return False
        salt = user["salt"]
        expected = user["password_hash"]
        provided = self._hash_password(password, salt)
        return hmac.compare_digest(expected, provided)

    def user_exists(self, username: str) -> bool:
        return username.strip().lower() in self._users


# Singleton store used by the Flask app
auth_store = InMemoryAuthStore()


