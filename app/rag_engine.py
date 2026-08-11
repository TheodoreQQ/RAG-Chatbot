import os 
import shutil 
from typing import List 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from app.config import CHROMA_PATH, UPLOAD_DIR, EMBEDDING_MODEL, LLM_MODEL
from app.models import ChatMessage

class RAGEngine():
  def __init__(self):
    self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    self.vector_store = Chroma(
      persist_directory=CHROMA_PATH,
      embedding_function=self.embeddings
    )
    
    self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
    self.llm = ChatOllama(model=LLM_MODEL, temperature=0.0, num_ctx=4096)
    
    self._setup_prompts()
    
  def _setup_prompts(self):
    contextualize_q_system_prompt = (
      "Mając daną historię konwersacji oraz najnowsze pytanie użytkownika, "
      "które może odwoływać się do kontekstu w historii, sformułuj samodzielne pytanie, "
      "które można zrozumieć bez znajomości historii konwersacji. "
      "NIE odpowiadaj na pytanie, po prostu zmień jego treść jeśli to konieczne, "
      "w przeciwnym razie zwróć je bez zmian."
    )
      
    self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
      ("system", contextualize_q_system_prompt),
      MessagesPlaceholder(variable_name="chat_history"),
      ("human", "{question}"),
    ])
      
    qa_system_prompt = (
      "Jesteś pomocnym asystentem. Odpowiadaj na pytania użytkownika WYŁĄCZNIE na podstawie poniższego kontekstu z dokumentów. "
      "Pod żadnym pozorem nie odpowiadaj na pytania przy użyciu innych języków niż POLSKI (np. Chiński  czy angielski). Chyba, że chodzi o nazwę własną lub fragment kodu - wtedy możesz"
      "Jeśli w kontekście nie ma odpowiedzi, napisz szczerze i tylko tyle: 'Nie jestem w stanie udzielić odpowiedzi na zadanie pytanie'.\n\n"
      "KONTEKST Z DOKUMENTÓW:\n"
      "{context}"
    )
      
    self.qa_prompt = ChatPromptTemplate.from_messages([
      ("system", qa_system_prompt),
      MessagesPlaceholder(variable_name="chat_history"),
      ("human", "{question}"),
    ])
      
      
  def _convert_history(self, history: List[ChatMessage]):
    lc_history = []
    for msg in history:
      if msg.role == "user":
        lc_history.append(HumanMessage(content=msg.content))
      elif msg.role == "assistant":
        lc_history.append(AIMessage(content=msg.content))
    return lc_history
    
  def generate_response(self, message: str, history: List[ChatMessage]) -> str:
    lc_history = self._convert_history(history)
      
    if lc_history:
      try:
        prompt_to_rephrase = self.contextualize_q_prompt.format_messages(
          chat_history=lc_history,
          question=message
        ) 
        
        standalone_question = self.llm.invoke(prompt_to_rephrase).content.strip()
      except Exception as ex:
        print(f"BŁĄD Query Rewriting: {ex}")
        standalone_question = message
    else:
      standalone_question = message
          
    try:  
      relevant_docs = self.retriever.invoke(standalone_question)
      context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    except Exception as ex:
      print(f"BŁĄD Retrieval: {ex}")
      context_text = ""
      
    if not context_text:
      return "Baza wiedzy jest pusta"
    
    formatted_prompt = self.qa_prompt.format_messages(
      context=context_text,
      chat_history=lc_history,
      question=message
    )
      
    response = self.llm.invoke(formatted_prompt)
    return response.content


  def add_document(self, file_path: str, file_name: str):
    loader = PyPDFLoader(file_path)
    documents =   loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
      
    for chunk in chunks:
      chunk.metadata["source"] = file_name
        
      
    self.vector_store.add_documents(chunks)
      
  def delete_document(self, file_name: str): 
    self.vector_store.delte(where={"source": file_name})
    
rag_engine = RAGEngine()
      
      