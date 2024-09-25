from fastapi import Request, HTTPException, status
from fastapi_sessions.frontends.implementations import SessionCookie
from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters

from fastapi_sessions.session_verifier import SessionVerifier
# from fastapi_sessions.backends.session_backend import SessionData
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from uuid import UUID, uuid4

class SessionData(BaseModel):
    username: str


cookie_params = CookieParameters()

# Uses UUID
cookie = SessionCookie(
    cookie_name="cookie",
    identifier="general_verifier",
    auto_error=True,
    secret_key="DONOTUSE",
    cookie_params=cookie_params,
)
backend = InMemoryBackend[UUID, SessionData]()


class BasicVerifier(SessionVerifier[UUID, SessionData]):
    def __init__(
        self,
        *,
        identifier: str,
        auto_error: bool,
        backend: InMemoryBackend[UUID, SessionData],
        auth_http_exception: HTTPException,
    ):
        self._identifier = identifier
        self._auto_error = auto_error
        self._backend = backend
        self._auth_http_exception = auth_http_exception

    @property
    def identifier(self):
        return self._identifier

    @property
    def backend(self):
        return self._backend

    @property
    def auto_error(self):
        return self._auto_error

    @property
    def auth_http_exception(self):
        return self._auth_http_exception

    def verify_session(self, model: SessionData) -> bool:
        """If the session exists, it is valid"""
        return True

# class CheckSessionMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next):
#         if request.url.path not in ["/api/login","/docs","/favicon.ico"]:
#             session_id = request.cookies.get("session_id")
#             if session_id:
#                 try:
#                     UUID(session_id)
#                 except ValueError:
#                     raise HTTPException(status_code=400, detail="Invalid session ID")
#             else:
#                 raise HTTPException(status_code=403, detail="Session ID missing")
            
#             response = await call_next(request)
#         return response

verifier = BasicVerifier(
    identifier="general_verifier",
    auto_error=True,
    backend=backend,
    auth_http_exception=HTTPException(status_code=403, detail="invalid session"),
)

async def get_session(request: Request) -> SessionData:
    session = await verifier(request)
    if session is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No active session")
    return session
