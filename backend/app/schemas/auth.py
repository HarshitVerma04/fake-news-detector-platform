"""
backend/app/schemas/auth.py

Pydantic schemas for authentication endpoints.
"""

from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    username: str  = Field(..., min_length=3, max_length=50)
    email:    EmailStr
    password: str  = Field(..., min_length=8, max_length=128)


class LoginResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    user:          "UserPublic"


class RefreshRequest(BaseModel):
    refresh_token: str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token:        str
    new_password: str = Field(..., min_length=8, max_length=128)


class UserPublic(BaseModel):
    id:         int
    username:   str
    email:      str
    created_at: datetime

    model_config = {"from_attributes": True}


class UserUpdateRequest(BaseModel):
    username: str | None = Field(None, min_length=3, max_length=50)
    email:    EmailStr | None = None


LoginResponse.model_rebuild()