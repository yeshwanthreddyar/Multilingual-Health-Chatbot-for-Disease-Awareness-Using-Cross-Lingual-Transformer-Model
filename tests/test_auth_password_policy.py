from __future__ import annotations

from app.api.auth import validate_password_policy
from app.api.auth import pwd_context


def test_long_password_allowed_with_argon2_only() -> None:
    """
    In the current configuration we use argon2 only, so very long passwords
    (e.g. ADMIN_PASSWORD) should not raise a bcrypt length error.
    """
    assert "bcrypt" not in set(pwd_context.schemes())
    very_long = "x" * 200
    # Should not raise
    validate_password_policy(very_long)

