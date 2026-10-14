import os
import gc
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from app.config import UPLOAD_DIR
from app.services.rag_service import rag_service

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
    
  clean_filename = os.path.basename(file.filename)
  os.makedirs(UPLOAD_DIR, exist_ok=True)
  file_path = os.path.join(UPLOAD_DIR, clean_filename)
    
  if os.path.exists(file_path):
    raise HTTPException(
      status_code=400,
      detail=f"Plik '{clean_filename}' już istnieje w bazie wiedzy! Usuń starą wersję przed aktualizacją."
    )
    
  # 1. Zapis pliku fizycznego na dysku
  try:
    with open(file_path, "wb") as buffer:
      shutil.copyfileobj(file.file, buffer)
  except Exception as ex:
    raise HTTPException(status_code=500, detail=f"Błąd zapisu pliku na dysku: {str(ex)}")
    
  # 2. Indeksowanie w ChromaDB (przez nasz wydzielony service)
  try: 
    rag_service.add_document(file_path, clean_filename)
  except ValueError as ve:
    if os.path.exists(file_path):
      os.remove(file_path)
    raise HTTPException(status_code=400, detail=str(ve))
  except Exception as ex:
    # Sprzątanie pliku z dysku w przypadku błędu
    if os.path.exists(file_path):
      os.remove(file_path)
    raise HTTPException(status_code=500, detail=f"Błąd indeksowania w bazie wiedzy: {str(ex)}")
    
  return {"status": "success", "message": f"Pomyślnie zaindeksowano plik: {file.filename}"}


@router.delete("/{filename}")
async def delete_document(filename: str):
  clean_filename = os.path.basename(filename)
  file_path = os.path.join(UPLOAD_DIR, clean_filename)
    
  # 1. Usuwanie wektorów z ChromaDB
  try:
    rag_service.delete_document(clean_filename)
  except Exception as e:
    print(f"[OSTRZEŻENIE ChromaDB] Błąd usuwania wektorów: {e}")

  # 2. Zwolnienie uchwytów (dla Windowsa)
  gc.collect()

  # 3. Usuwanie pliku z dysku
  if os.path.exists(file_path):
    try:
      os.remove(file_path)
    except Exception as e:
      raise HTTPException(
        status_code=500, 
        detail=f"Plik na dysku jest zablokowany: {str(e)}"
      )

  return {"status": "success", "message": f"Usunięto plik {clean_filename}"}
    
  