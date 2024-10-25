import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import Field
import pydash


router = APIRouter()



@router.get("/download_file", response_description="Download file using the file path")
async def download_file(file_path: str):
    # Validate the file path
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found or invalid file path")

    # Return the file as a response
    return FileResponse(path=file_path, filename=os.path.basename(file_path))