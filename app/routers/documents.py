import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import UPLOAD_DIR
from app.rag_engine import rag_engine

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("")
async def upload_documents(file: UploadFile = File(...)):
  if not file.filename.endswith(".pdf"):
    raise HTTPException(status_code=400, detail="Tylko obsługa plików PDF.")
  
  file_path = os.path.join(UPLOAD_DIR, file.filename)
  with open(file_path, "wb") as buffer:
    shutil.copyfileobj(file.file, buffer)
  
  rag_engine.add_document(file_path, file.filename)
  return {"status": "success", "message": f"Zaindeksowano plik {file.filename}"}

@router.get("")
async def list_documents():
  files = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
  return {"Documents": files}

@router.delete("/{filename}")
async def delete_document(file_name: str):
  file_path = os.path.join(UPLOAD_DIR, file_name)
  
  try:
    rag_engine.delete_document(file_name)
  except Exception as ex:
    raise HTTPException(status_code=500, detail=f"Błąd usuwania z bazy: {str(ex)}")
  
  if os.path.exists(file_path):
    os.remove(file_path)
    
  return {"status": "success", "message": f"Usunięto plik {file_name}"}
    
  