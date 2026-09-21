import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentLoader:
  def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
    self.text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap
    )

  def load_and_split(self, file_path: str, file_name: str) -> List[Document]:
    clean_filename = os.path.basename(file_name)
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
        
    chunks = self.text_splitter.split_documents(documents)
        
    # Stosujemy jednolite metadane 'source'
    for chunk in chunks:
      chunk.metadata["source"] = clean_filename
            
    return chunks