import logging
from fastapi import FastAPI, Depends, HTTPException , Path, Query , Response
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from app.database import SessionLocal, get_db
from app import database
from app.model import model_llm
from app.schema import schema_llm
from fastapi import APIRouter
from uuid import UUID, uuid4
from app.endpoint.login import cookie , SessionData , verifier
from app.endpoint import sendMail
from passlib.context import CryptContext
from app.utils import utils 
import pydash


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()
# app = FastAPI()



# CRUD operations for Users
@router.post("/users", response_model=schema_llm.User)
def create_user(user: schema_llm.UserCreate, db: Session = Depends(get_db) ):
    # hashed_password = pwd_context.hash(user.password.get_secret_value())
    hashed_password = utils.hash_password(user.password.get_secret_value())
    db_user = model_llm.User(
        user_id=user.user_id, 
        password=hashed_password,  
        nickname=user.nickname
        # user_roll=user.user_roll
    )
    db.add(db_user)
    db.flush()
    db.refresh(db_user)
    return db_user

@router.get("/users/{user_id}", response_model=schema_llm.User, description='특정 사용자를 검색한다.' )
def read_user(user_id: str = Path(..., description='검색할사용자ID'), db: Session = Depends(get_db)  ):
    db_user = db.query(model_llm.User).filter(model_llm.db_comment_endpoint).filter(model_llm.User.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.get("/users/nickname/{nickname}", response_model=schema_llm.User)
def read_nickname(nickname: str, db: Session = Depends(get_db) ):
    db_user = db.query(model_llm.User).filter(model_llm.db_comment_endpoint).filter(model_llm.User.nickname == nickname).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="NickName not found")
    return db_user

@router.get("/read_all_users", response_model=List[schema_llm.User], dependencies=[Depends(cookie)])
def read_all_users(db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):

    query = db.query(model_llm.User)
    users = query.all()
    return users

#search users
@router.post(
    "/search_users", 
    response_model=List[schema_llm.User],  
    description='''<pre>
    
    user_search_field(선택)
        nickname, email

    search_words(선택)

    user_roll(선택)
        OWNER	    오너
        MANAGER	    관리자
        DEVELOPER	개발자
        GUEST	    게스트

    페이징처리(선택): 아래이 항목이 있으면 페이징 처리됨. 없으면 모두 나옴.
        offset, limit
</pre>''', 
    dependencies=[Depends(cookie)], 
)
def search_users(search: schema_llm.UserSearch, db: Session = Depends(get_db), session_data: SessionData = Depends(verifier)):
    query = db.query(model_llm.User)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 

    search_exclude = search.dict(exclude_unset=True)

    if (
        'search_words' in search_exclude and 
        search_exclude['search_words'] != "" and 
        'user_search_field' in search_exclude  and 
        search_exclude['user_search_field'] != "" 
    ) :
        if(search.user_search_field == 'nickname'): 
            query = query.filter(model_llm.User.nickname.like(f'%{search.search_words}%'))
        if(search.user_search_field == 'email'): 
            query = query.filter(model_llm.User.user_id.like(f'%{search.search_words}%'))

    if 'user_roll' in search_exclude and len(search_exclude['user_roll']):
        query = query.filter(model_llm.User.user_roll == search_exclude['user_roll'])

    # Apply pagination
    if 'skip' in search_exclude and 'limit' in search_exclude :
        query = query.offset(search_exclude['skip']).limit(search_exclude['limit'])

    users = query.all()
    return users

@router.put(
    "/users/{user_id}", 
    response_model=schema_llm.User, 
    dependencies=[Depends(cookie)]
)
def update_user(
    user_id: str, 
    user: schema_llm.UserUpdate, 
    db: Session = Depends(get_db) ,
    session_data: SessionData = Depends(verifier)
):
    db_user = db.query(model_llm.User).filter(model_llm.db_comment_endpoint).filter(model_llm.User.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # db_user.nickname = user.nickname
    # db_user.
    update_data = user.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if key == "password":
            hashed_password = utils.hash_password(value.get_secret_value())
            setattr(db_user, key, hashed_password)
        else:
            setattr(db_user, key, value)
    # db_user.update_user = session_data.user_id

    db.flush()
    db.refresh(db_user)
    return db_user

#search users
@router.post(
    "/update_user_role", 
    response_model=List[schema_llm.User],  
    description='''<pre>

    사용자 역할을 바꾸는 API   
     
    user_id(필수)
    user_roll(필수)
        OWNER	    오너
        MANAGER	    관리자
        DEVELOPER	개발자
        GUEST	    게스트

</pre>''', 
    dependencies=[Depends(cookie)], 
)
def update_user_role(users: schema_llm.UserRoleUpdates ,db: Session = Depends(get_db), session_data: SessionData = Depends(verifier)):
    updated_users = []

    for user_update in users.user_list:
        db_user = db.query(model_llm.User).filter(model_llm.db_comment_endpoint).filter(model_llm.User.user_id == user_update.user_id).first()
        if db_user is not None:

            update_data = user_update.dict(exclude_unset=True)
            for key, value in update_data.items():
                if key == "password":
                    hashed_password = utils.hash_password(value.get_secret_value())
                    setattr(db_user, key, hashed_password)
                else:
                    setattr(db_user, key, value)

            # Optionally set the user who made the update
            # db_user.updated_by = session_data.user_id
            
            updated_users.append(db_user)
    
    db.flush()
    for user in updated_users:
        db.refresh(user)
        
    return updated_users

@router.post("/user/compare_password", tags=["Users"])
async def compare_password(user: schema_llm.UserLogin, response: Response, db: Session = Depends(get_db) ):
    db_user = db.query(model_llm.User).filter(
        model_llm.User.user_id == user.user_id
        # , model_llm.User.password == hashed_password
    ).filter(model_llm.db_comment_endpoint).first()

    # Check if the user exists and verify the password
    if db_user is None or not utils.verify_password(user.password.get_secret_value(), db_user.password):
        raise HTTPException(status_code=401, detail="Invalid user_id or password")
    
    # return f"created session for {user_id}"
    return {
        "message": "password is correct"
    }

@router.delete("/users/{user_id}", response_model=schema_llm.User, dependencies=[Depends(cookie)],tags=["Users"])
def delete_user(user_id: str, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    db_user = db.query(model_llm.User).filter(model_llm.db_comment_endpoint).filter(model_llm.User.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Convert to Pydantic model before deleting
    user_data = User.from_orm(db_user)
    db.delete(db_user)
    db.flush()
    return user_data

@router.put(
    "/user/{user_id}/tokens", 
    response_model=schema_llm.User, 
    dependencies=[Depends(cookie)],
    description="""
    <pre>
    <h3>Request Body:</h3>
    <ul>
        <li>token_gitlab (Optional[str]): GitLab Access Token</li>
        <li>token_confluence (Optional[str]): Confluence Access Token</li>
        <li>token_jira (Optional[str]): Jira Access Token</li>
    </ul>
    </pre>
    """
)
def update_user_tokens(
    user_id: str, 
    token_data: schema_llm.UserUpdateToken, 
    db: Session = Depends(get_db), 
    session_data: SessionData = Depends(verifier)
):
    user = db.query(model_llm.User).filter(model_llm.db_comment_endpoint).filter(model_llm.User.user_id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    for key, value in token_data.dict(exclude_unset=True).items():
        setattr(user, key, value)
    # user.updated_at = lambda: datetime.now(KST)()
    
    db.flush()
    db.refresh(user)
    
    return user


@router.put(
    "/users/password_init/{user_id}",
    # dependencies=[Depends(cookie)],
    description="""
    비밀번호 초기화
    
    사용자아이디(user_id) 로 발송됨.
    사용자아이디가 정확한 email 이어야 함.
    
    내부발송
        sk.com메일도메인은SKT 내부구성원메일발송시에만사용하며, 
        수신자메일주소가외부메일일경우발송불가
        ex) mgs@sk.com(발신자메일주소) -> xxxxxxxxx@sk.com(수신자메일주소)
    외부발송
        메일도메인을sktelecom.com으로지정후메일발송
        ex) mgs@sktelecom.com(발신자메일주소) -> xxxxxx@naver.com(수신자메일주소)
    """
)
async def password_init(
    user_id: str, 
    db: Session = Depends(get_db)
):
    logger.info("Start Password initialize")
    db_user = db.query(model_llm.User).filter(model_llm.db_comment_endpoint).filter(model_llm.User.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    init_password = "Sktelecom1!"
    init_password = utils.generate_random_password()
    db_user.password = utils.hash_password(init_password)
    
    mail_data = schema_llm.EmailRequest(
        receiver_email=user_id,
        subject="Password for daisy is initialized",
        content=f"intialized password is  {init_password}"
    )
    
    logger.info(f"mail data : {mail_data}")
    
    mail_response = await sendMail.send_email(email_request=mail_data)
    
    return {
        "message": f"Password is initialized. Please confirm email {user_id} "
    }
