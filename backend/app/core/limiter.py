"""
backend/app/core/limiter.py

Rate limiting using slowapi (built on limits library).
Limits are applied per IP address.

Current limits:
  - /predict and /fetch-url : 30 requests per minute
  - /auth/register          : 5 requests per minute
  - /auth/login             : 10 requests per minute
  - /auth/password-reset    : 3 requests per minute
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
