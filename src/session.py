"""
Session manager.
Tracks all active WebSocket sessions.
Each session has:
  - session_id
  - user_a: { websocket, language }
  - user_b: { websocket, language }
  - created_at, last_active
  - audio queues per user
  - transcript queues per user
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict
from fastapi import WebSocket


@dataclass
class UserState:
    websocket:  WebSocket
    language:   str = "en"
    audio_q:    asyncio.Queue = field(default_factory=asyncio.Queue)
    result_q:   asyncio.Queue = field(default_factory=asyncio.Queue)


@dataclass
class Session:
    session_id:   str
    created_at:   float = field(default_factory=time.time)
    last_active:  float = field(default_factory=time.time)
    users:        Dict[str, UserState] = field(default_factory=dict)

    def touch(self):
        self.last_active = time.time()

    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        return time.time() - self.last_active > ttl_seconds

    def add_user(self, user_id: str, websocket: WebSocket, language: str):
        self.users[user_id] = UserState(
            websocket=websocket,
            language=language
        )
        self.touch()

    def remove_user(self, user_id: str, websocket: Optional[WebSocket] = None):
        
        if user_id in self.users:
            if websocket is None or self.users[user_id].websocket is websocket:
                del self.users[user_id]

    def get_other_users(self, user_id: str) -> list:
        return [uid for uid in self.users if uid != user_id]


class SessionManager:
    """
    Manages all active translation sessions.
    Sessions expire after 1 hour of inactivity.
    """

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        print("[SessionManager] Ready.")

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())[:8]
        self._sessions[session_id] = Session(session_id=session_id)
        print(f"[SessionManager] Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def remove_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]
            print(f"[SessionManager] Removed session: {session_id}")

    def cleanup_expired(self):
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            self.remove_session(sid)
        if expired:
            print(f"[SessionManager] Cleaned up {len(expired)} expired session(s).")

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# singleton
session_manager = SessionManager()