from pydantic import BaseModel
from fastapi import HTTPException, APIRouter, FastAPI,Request,  Response, Depends
from uuid import UUID, uuid4

from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.session_verifier import SessionVerifier
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters
from sqlalchemy import text
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List, Optional
from app.database import SessionLocal, get_db
from app import models, database
from sqlalchemy.orm import Session
from app.utils.utils import pwd_context , hash_password ,  verify_password

router = APIRouter()



class SessionData(BaseModel):
    user_id: str
    nickname: str
    user_roll: str
    token_gitlab: Optional[str]
    token_confluence: Optional[str]
    token_jira: Optional[str]



cookie_params = CookieParameters()

# Uses UUID
cookie = SessionCookie(
    cookie_name="cookie_daisy",
    identifier="general_verifier",
    auto_error=True,
    secret_key="DONOTUSE",
    cookie_params=cookie_params,
)
session_backend = InMemoryBackend[UUID, SessionData]()


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

 
verifier = BasicVerifier(
    identifier="general_verifier",
    auto_error=True,
    backend=session_backend,
    auth_http_exception=HTTPException(status_code=403, detail="invalid session"),
)


class CheckSessionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app , backend: InMemoryBackend , cookie, exempt_paths: List[str] = None):
        super().__init__(app)
        if exempt_paths is None:
            exempt_paths = []
        self.exempt_paths = exempt_paths
        self.backend = backend
        self.cookie = cookie
        

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        session_id = request.cookies.get("cookie")
        if session_id:
            try:
                uuid_session_id = UUID(session_id)
                session_data = await self.backend.read(uuid_session_id)
                # session_data = await backend.read(session_id)
                if not session_data:
                    raise HTTPException(status_code=403, detail="Invalid session")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid session ID")
        else:
            raise HTTPException(status_code=403, detail="Session ID missing")
        
        # try:
        #     # session_data = await self.backend.read(self.cookie)
        #     # if not session_data:
        #     if not self.cookie: 
        #         raise HTTPException(status_code=403, detail="Invalid session")
        # except ValueError:
        #     raise HTTPException(status_code=400, detail="Invalid session ID")

        response = await call_next(request)
        return response
   

@router.post("/create_session", tags=["Session"])
async def create_session(login: models.UserLogin, response: Response, db: Session = Depends(get_db) ):
    query = db.query(database.User)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 
    query = query.filter(
        database.User.user_id == login.user_id 
        # , database.User.password == hashed_password
    )
    db_user = query.first()

    # Check if the user exists and verify the password
    if db_user is None or not verify_password(login.password.get_secret_value(), db_user.password):
        raise HTTPException(status_code=404, detail="Invalid user_id or password")
    
    
    session = uuid4()
    data = SessionData(
        user_id=login.user_id,
        nickname=getattr(db_user, 'nickname'),
        user_roll=getattr(db_user, 'user_roll'),
        token_gitlab=getattr(db_user, 'token_gitlab'),
        token_confluence=getattr(db_user, 'token_confluence'),
        token_jira=getattr(db_user, 'token_jira')
    )

    await session_backend.create(session, data)
    cookie.attach_to_response(response, session)
    
    # return f"created session for {user_id}"
    return {
        "message": f"created session for {login.user_id}",
        "session_data" : data
    }


@router.get("/whoami", dependencies=[Depends(cookie)], tags=["Session"])
async def whoami(session_data: SessionData = Depends(verifier)):
    return session_data


@router.post("/delete_session", dependencies=[Depends(cookie)], tags=["Session"])
async def del_session(response: Response, session_id: UUID = Depends(cookie)):
    await session_backend.delete(session_id)
    cookie.delete_from_response(response)
    return "deleted session"