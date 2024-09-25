from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from .models import User
from .database import SessionLocal, engine, Base
from typing import Dict , List , Any
from sqlalchemy.inspection import inspect
from fastapi import APIRouter

router = APIRouter()
# app = FastAPI()

# Create tables
# Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/create_user/", response_model=int , tags=["Sample"])
def create_user(name: str, age: int, db: Session = Depends(get_db)):
    db_user = User(name=name, age=age)
    db.add(db_user)
    db.commit()
    return db_user.id

@router.get("/users/{user_id}", response_model=str , tags=["Sample"])
def users(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return f"User: {db_user.name}, Age: {db_user.age}"

# Define the Pydantic model for the response
class UserResponse(BaseModel):
    name: str
    age: int

@router.get("/users1/{user_id}", response_model=UserResponse , tags=["Sample"])
def users1(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"name": db_user.name, "age": db_user.age}

@router.get("/users2/{user_id}", response_model=Dict , tags=["Sample"])
def users2(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    # return {"name": db_user.name, "age": db_user.age}
    # user_dict = {c.name: getattr(db_user, c.name) for c in db_user.__table__.columns}

    mapper = inspect(db_user)
    user_dict = {attr.key: getattr(db_user, attr.key) for attr in mapper.attrs if not attr.key.startswith('_')}
    return user_dict


# Endpoint to get a list of all users
@router.get("/userList/", response_model=List[Dict] , tags=["Sample"])
def userList(db: Session = Depends(get_db)):
    db_users = db.query(User).all()  # Adjust this query as needed
    # users_list = [
    #     {c.name: getattr(user, c.name) for c in user.__table__.columns}
    #     for user in db_users
    # ]

    users_list = []
    for db_user in db_users :
        mapper = inspect(db_user)
        user_dict = {attr.key: getattr(db_user, attr.key) for attr in mapper.attrs if not attr.key.startswith('_')}
        users_list.append(user_dict)
    return users_list


class SearchUser(BaseModel):
    name: str
    age: int

@router.get("/listPayoadTypeOfSearchUser/", response_model=List[Dict] , tags=["Sample"] )
def listPayoadTypeOfSearchUser(user: SearchUser, db: Session = Depends(get_db)):
    db_users = db.query(User).all()  # Adjust this query as needed
    # users_list = [
    #     {c.name: getattr(user, c.name) for c in user.__table__.columns}
    #     for user in db_users
    # ]

    users_list = []
    for db_user in db_users :
        mapper = inspect(db_user)
        user_dict = {attr.key: getattr(db_user, attr.key) for attr in mapper.attrs if not attr.key.startswith('_')}
        users_list.append(user_dict)
    return users_list


class SearchPayload(BaseModel):
    name: str
    age: int
    hobby: List

@router.get("/listSearchPayload/", response_model=List[Any], tags=["Sample"])
def listSearchPayload(payload: SearchPayload, db: Session = Depends(get_db)):
    # Filter by name and check if the hobby is in the provided list
    db_users = db.query(User).filter(User.name == payload.name, User.hobby.in_(payload.hobby)).all()

    users_list = []
    for db_user in db_users:
        mapper = inspect(db_user)
        user_dict = {attr.key: getattr(db_user, attr.key) for attr in mapper.attrs if not attr.key.startswith('_')}
        users_list.append(user_dict)
    return users_list

