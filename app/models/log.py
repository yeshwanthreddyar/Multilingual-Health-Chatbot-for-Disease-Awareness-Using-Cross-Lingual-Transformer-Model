"""
Log model - user prompts and responses for admin and training.
"""
from __future__ import annotations

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.sql import func
from app.db import Base


class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=True, index=True)
    session_id = Column(String(100), nullable=True)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    intent = Column(String(100), nullable=True)
    is_health_related = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
