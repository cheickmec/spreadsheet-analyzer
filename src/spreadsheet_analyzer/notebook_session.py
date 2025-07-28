"""
Notebook Session Management

Handles lifecycle of notebook sessions with proper async context management.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from result import Err, Ok, Result

from spreadsheet_analyzer.notebook_tools import NotebookToolkit, create_toolkit


class NotebookSession:
    """Manages a notebook session with proper lifecycle."""

    def __init__(self, toolkit: NotebookToolkit):
        self.toolkit = toolkit
        self._is_active = False

    async def start(self) -> Result[None, str]:
        """Start the notebook session."""
        try:
            await self.toolkit.kernel_service.create_session(self.toolkit.session_id)
            self._is_active = True
            return Ok(None)
        except Exception as e:
            return Err(f"Failed to start session: {e!s}")

    async def stop(self) -> Result[None, str]:
        """Stop the notebook session."""
        try:
            if self._is_active:
                await self.toolkit.kernel_service.close_session(self.toolkit.session_id)
                self._is_active = False
            return Ok(None)
        except Exception as e:
            return Err(f"Failed to stop session: {e!s}")

    async def execute(self, code: str, cell_id: str | None = None) -> Result[Any, str]:
        """Execute code in the session."""
        if not self._is_active:
            return Err("Session not active")

        result = await self.toolkit.execute_code(code, cell_id)
        return result

    def is_active(self) -> bool:
        """Check if session is active."""
        return self._is_active

    def get_session_id(self) -> str:
        """Get the session ID."""
        return self.toolkit.session_id


@asynccontextmanager
async def notebook_session(session_id: str, notebook_path: Path | None = None):
    """Async context manager for notebook sessions."""
    # Create kernel service with proper initialization
    from spreadsheet_analyzer.core_exec import KernelProfile, KernelService

    kernel_service = KernelService(KernelProfile())

    async with kernel_service:
        toolkit = create_toolkit(kernel_service, session_id, notebook_path)
        session = NotebookSession(toolkit)

        try:
            # Start session
            start_result = await session.start()
            if start_result.is_err():
                raise RuntimeError(f"Failed to start session: {start_result.unwrap_err()}")

            yield session

        finally:
            # Stop session
            await session.stop()


class SessionManager:
    """Manages multiple notebook sessions."""

    def __init__(self):
        self._sessions: dict[str, NotebookSession] = {}

    async def create_session(self, session_id: str, kernel_name: str = "python3") -> Result[NotebookSession, str]:
        """Create a new notebook session."""
        if session_id in self._sessions:
            return Err(f"Session {session_id} already exists")

        toolkit_result = create_toolkit(kernel_name, session_id)
        if toolkit_result.is_err():
            return toolkit_result

        toolkit = toolkit_result.unwrap()
        session = NotebookSession(toolkit)

        start_result = await session.start()
        if start_result.is_err():
            return start_result

        self._sessions[session_id] = session
        return Ok(session)

    async def get_session(self, session_id: str) -> Result[NotebookSession, str]:
        """Get an existing session."""
        if session_id not in self._sessions:
            return Err(f"Session {session_id} not found")
        return Ok(self._sessions[session_id])

    async def close_session(self, session_id: str) -> Result[None, str]:
        """Close a session."""
        if session_id not in self._sessions:
            return Err(f"Session {session_id} not found")

        session = self._sessions[session_id]
        await session.stop()
        del self._sessions[session_id]
        return Ok(None)

    async def close_all(self) -> None:
        """Close all sessions."""
        for session_id in list(self._sessions.keys()):
            await self.close_session(session_id)

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())


# Global session manager instance
_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    """Get the global session manager."""
    return _session_manager
