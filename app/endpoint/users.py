from fastapi import FastAPI, Depends, HTTPException , Path, Query , Response
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from app.models import User , UserCreate , UserUpdate
from app.database import SessionLocal, get_db
from app import models, database
from fastapi import APIRouter
from uuid import UUID, uuid4
from app.endpoint.login import cookie , SessionData , verifier
from passlib.context import CryptContext
from app.utils.utils import pwd_context , hash_password ,  verify_password
import pydash

router = APIRouter()
# app = FastAPI()



# CRUD operations for Users
@router.post("/users", response_model=User, tags=["Users"])
def create_user(user: UserCreate, db: Session = Depends(get_db) ):
    # hashed_password = pwd_context.hash(user.password.get_secret_value())
    hashed_password = hash_password(user.password.get_secret_value())
    db_user = database.User(
        user_id=user.user_id, 
        password=hashed_password,  
        nickname=user.nickname
        # user_roll=user.user_roll
    )
    db.add(db_user)
    db.flush()
    db.refresh(db_user)
    return db_user

@router.get("/users/{user_id}", response_model=User, description='특정 사용자를 검색한다.' , tags=["Users"])
def read_user(user_id: str = Path(..., description='검색할사용자ID'), db: Session = Depends(get_db)  ):
    db_user = db.query(database.User).filter(database.db_comment_endpoint).filter(database.User.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.get("/users/nickname/{nickname}", response_model=User ,tags=["Users"])
def read_nickname(nickname: str, db: Session = Depends(get_db) ):
    db_user = db.query(database.User).filter(database.db_comment_endpoint).filter(database.User.nickname == nickname).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="NickName not found")
    return db_user

@router.get("/read_all_users", response_model=List[User], dependencies=[Depends(cookie)],tags=["Users"])
def read_all_users(db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):

    query = db.query(database.User)
    users = query.all()
    return users

#search users
@router.post(
    "/search_users", 
    response_model=List[User],  
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
    tags=["Users"]
)
def search_users(search: models.UserSearch, db: Session = Depends(get_db), session_data: SessionData = Depends(verifier)):
    query = db.query(database.User)
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
            query = query.filter(database.User.nickname.like(f'%{search.search_words}%'))
        if(search.user_search_field == 'email'): 
            query = query.filter(database.User.user_id.like(f'%{search.search_words}%'))

    if 'user_roll' in search_exclude and len(search_exclude['user_roll']):
        query = query.filter(database.User.user_roll == search_exclude['user_roll'])

    # Apply pagination
    if 'skip' in search_exclude and 'limit' in search_exclude :
        query = query.offset(search_exclude['skip']).limit(search_exclude['limit'])

    users = query.all()
    return users

@router.put(
    "/users/{user_id}", 
    response_model=User, 
    # dependencies=[Depends(cookie)],
    tags=["Users"])
def update_user(user_id: str, user: UserUpdate, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    db_user = db.query(database.User).filter(database.db_comment_endpoint).filter(database.User.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # db_user.nickname = user.nickname
    # db_user.
    update_data = user.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if key == "password":
            hashed_password = hash_password(value.get_secret_value())
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
    response_model=List[User],  
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
    tags=["Users"]
)
def update_user_role(users: models.UserRoleUpdates ,db: Session = Depends(get_db), session_data: SessionData = Depends(verifier)):
    updated_users = []

    for user_update in users.user_list:
        db_user = db.query(database.User).filter(database.db_comment_endpoint).filter(database.User.user_id == user_update.user_id).first()
        if db_user is not None:

            update_data = user_update.dict(exclude_unset=True)
            for key, value in update_data.items():
                if key == "password":
                    hashed_password = hash_password(value.get_secret_value())
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
async def compare_password(user: models.UserLogin, response: Response, db: Session = Depends(get_db) ):
    db_user = db.query(database.User).filter(
        database.User.user_id == user.user_id
        # , database.User.password == hashed_password
    ).filter(database.db_comment_endpoint).first()

    # Check if the user exists and verify the password
    if db_user is None or not verify_password(user.password.get_secret_value(), db_user.password):
        raise HTTPException(status_code=401, detail="Invalid user_id or password")
    
    # return f"created session for {user_id}"
    return {
        "message": "password is correct"
    }

@router.delete("/users/{user_id}", response_model=User, dependencies=[Depends(cookie)],tags=["Users"])
def delete_user(user_id: str, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    db_user = db.query(database.User).filter(database.db_comment_endpoint).filter(database.User.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Convert to Pydantic model before deleting
    user_data = User.from_orm(db_user)
    db.delete(db_user)
    db.flush()
    return user_data

@router.put(
    "/user/{user_id}/tokens", 
    response_model=User, 
    dependencies=[Depends(cookie)],
    tags=["Users"],
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
    token_data: models.UserUpdateToken, 
    db: Session = Depends(get_db), 
    session_data: SessionData = Depends(verifier)
):
    user = db.query(database.User).filter(database.db_comment_endpoint).filter(database.User.user_id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    for key, value in token_data.dict(exclude_unset=True).items():
        setattr(user, key, value)
    # user.updated_at = lambda: datetime.now(KST)()
    
    db.flush()
    db.refresh(user)
    
    return user
