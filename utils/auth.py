import hashlib
import hmac
import json
import os
import threading
from typing import Dict, Optional


class InMemoryAuthStore:
    """Simple in-memory user store with salted password hashing."""

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


class PersistentAuthStore(InMemoryAuthStore):
    """File-backed auth store that persists users across restarts.

    Uses a JSON file to store { username: { salt, password_hash } }.
    """

    def __init__(self, filepath: Optional[str] = None) -> None:
        super().__init__()
        self._filepath = filepath or os.environ.get('AUTH_STORE_FILE', 'users_store.json')
        self._lock = threading.Lock()
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        try:
            if os.path.exists(self._filepath):
                with open(self._filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Basic validation
                    if isinstance(data, dict):
                        self._users = {
                            str(k).strip().lower(): {
                                'salt': v.get('salt', ''),
                                'password_hash': v.get('password_hash', '')
                            }
                            for k, v in data.items()
                            if isinstance(v, dict) and 'salt' in v and 'password_hash' in v
                        }
        except Exception:
            # If file is corrupt, keep in-memory empty store
            self._users = {}

    def _atomic_write(self, path: str, content: str) -> None:
        temp_path = f"{path}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(temp_path, path)

    def _save_to_disk(self) -> None:
        try:
            with self._lock:
                payload = json.dumps(self._users, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
                # Ensure parent directory exists
                parent = os.path.dirname(os.path.abspath(self._filepath))
                if parent and not os.path.exists(parent):
                    os.makedirs(parent, exist_ok=True)
                self._atomic_write(self._filepath, payload)
        except Exception:
            # Fail silently to avoid breaking auth flow on FS errors
            pass

    def create_user(self, username: str, password: str) -> Optional[str]:
        error = super().create_user(username, password)
        if error is None:
            self._save_to_disk()
        return error


# Singleton store used by the Flask app (file-backed persistence)
auth_store = PersistentAuthStore()


