import logging
from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app import models, database
from app.utils import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()

# CRUD operations for user alerts
@router.post("", response_model=models.User)
def create_user_alert(user: models.UserAlertCreate, db: Session = Depends(get_db)):
    logger.info(f"Creating user alert started. Info: {user}")
    # Always set user_roll to ALERT
    hashed_password = utils.hash_password(user.user_id)
    db_user = database.User(
        user_id=user.user_id,
        password=hashed_password,
        nickname=user.user_id,
        user_roll="ALERT",  # Always ALERT
        llm_api_id=user.llm_api_id,
        llm_model=user.llm_model
    )
    db.add(db_user)
    db.flush()
    db.refresh(db_user)
    logger.info(f"User alert created successfully. ID: {db_user.user_id}")
    return db_user

@router.get("/{user_id}", response_model=models.User, description="Retrieve user alert by user_id")
def read_user_alert(user_id: str = Path(..., description="User ID to retrieve"), db: Session = Depends(get_db)):
    logger.info(f"Retrieving user alert started for user_id: {user_id}")
    db_user = db.query(database.User).filter(database.User.user_id == user_id, database.User.user_roll == "ALERT").first()
    if db_user is None:
        logger.warning(f"User alert not found for user_id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found or not an ALERT user")
    logger.info(f"User alert retrieved successfully. Info: {db_user}")
    return db_user

@router.get("", response_model=List[models.User], description="Retrieve all users with user_roll ALERT")
def read_all_user_alerts(db: Session = Depends(get_db)):
    logger.info("Retrieving all user alerts started.")
    users = db.query(database.User).filter(database.User.user_roll == "ALERT").all()
    logger.info(f"Retrieved {len(users)} user alerts successfully.")
    return users

@router.put("/{user_id}", response_model=models.User, description="Update user alert details")
def update_user_alert(
    user_id: str,
    user: models.UserAlertUpdate,
    db: Session = Depends(get_db)
):
    logger.info(f"Updating user alert started for user_id: {user_id}. Update info: {user}")
    db_user = db.query(database.User).filter(database.User.user_id == user_id, database.User.user_roll == "ALERT").first()
    if db_user is None:
        logger.warning(f"User alert not found for update. User_id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found or not an ALERT user")
    
    update_data = user.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_user, key, value)
    
    db.flush()
    db.refresh(db_user)
    logger.info(f"User alert updated successfully. Info: {db_user}")
    return db_user

@router.delete("/{user_id}", response_model=models.User, description="Delete user alert")
def delete_user_alert(user_id: str, db: Session = Depends(get_db)):
    logger.info(f"Deleting user alert started for user_id: {user_id}")
    db_user = db.query(database.User).filter(database.User.user_id == user_id, database.User.user_roll == "ALERT").first()
    if db_user is None:
        logger.warning(f"User alert not found for deletion. User_id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found or not an ALERT user")
    
    user_data = models.User.from_orm(db_user)
    db.delete(db_user)
    db.flush()
    logger.info(f"User alert deleted successfully. ID: {user_id}")
    return user_data
