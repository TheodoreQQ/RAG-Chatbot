import os
import gc
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from app.config import UPLOAD_DIR
from app.rag_engine import rag_engine

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.get("")
async def list_documents():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.pdf')]
    return {"documents": files}

@router.post("")
async def upload_documents(file: UploadFile = File(...)):
  if not file.filename.endswith(".pdf"):
    raise HTTPException(status_code=400, detail="Tylko obsługa plików PDF.")
  
  os.makedirs(UPLOAD_DIR, exist_ok=True)
  file_path = os.path.join(UPLOAD_DIR, file.filename)
  
  if os.path.exists(file_path):
    raise HTTPException(
      status_code=400,
      detail=f"Plik '{file.filename}' już istnieje w bazie wiedzy! Usuń starą wersję przez aktualizacją! "
    )
  
  try:
    with open(file_path, "wb") as buffer:
      shutil.copyfileobj(file.file, buffer)
  except Exception as ex:
    raise HTTPException(status_code=500, detail=f"Błąd zapisu pliku: {str(ex)}")
  
  try: 
    rag_engine.add_document(file_path, file.filename)
  except Exception as ex:
    if os.path.exists(file_path):
      os.remove(file_path)
    raise HTTPException(status_code=500, detail=f"Błąd indeksowania w ChromaDB: {str(ex)}")
  
  return {"status": "success", "message": f"Zaindeksowano plik {file.filename}"}

@router.delete("/{file_name:path}")
async def delete_document(file_name: str):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    
    # 1. Usuwanie wektorów z ChromaDB
    try:
        rag_engine.delete_document(file_name)
    except Exception as e:
        print(f"[OSTRZEŻENIE ChromaDB] {e}")

    # 2. Zwolnienie uchwytów pliku
    gc.collect()

    # 3. Usuwanie fizycznego pliku z dysku
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Plik w bazie zablokowany przez system: {str(e)}"
            )

    return {"status": "success", "message": f"Usunięto plik {file_name}"}
    
  