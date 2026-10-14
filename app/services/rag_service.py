import os
from typing import List, Generator
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from app.config import LLM_MODEL
from app.models import ChatMessage
from app.core.document_loader import DocumentLoader
from app.core.vector_store import VectorStoreManager
from app.core.reranker import Reranker
from fastapi.responses import StreamingResponse

ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
class RAGService:
    def __init__(self):
        self.loader = DocumentLoader()
        self.vector_store = VectorStoreManager()
        self.reranker = Reranker()
        self.llm = ChatOllama(
            model=LLM_MODEL,
            base_url=ollama_url,
            temperature=0.0,
            num_ctx=4096)
        
        self._setup_prompts()

    def _setup_prompts(self):
        contextualize_q_system_prompt = (
            "Mając daną historię konwersacji oraz najnowsze pytanie użytkownika, "
            "które może odwoływać się do kontekstu w historii, sformułuj samodzielne pytanie, "
            "które można zrozumieć bez znajomości historii konwersacji.\n\n"
            "ZASADY:\n"
            "1. NIE odpowiadaj na pytanie.\n"
            "2. Zmień jego treść tylko wtedy, gdy odnosi się do historii, w przeciwnym razie zwróć je bez zmian.\n"
            "3. Wynikowe pytanie MUSI być w języku polskim."
            )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        qa_system_prompt = (
            "Jesteś pomocnym asystentem bazy wiedzy. Odpowiadaj na pytania użytkownika w języku polskim, "
            "wykorzystując wyłącznie informacje zawarte w poniższym KONTEKŚCIE Z DOKUMENTÓW.\n\n"
            "WYTYCZNE:\n"
            "1. Sformułuj zwięzłą, rzeczową odpowiedź (maksymalnie 3-5 zdań) na podstawie podanych fragmentów.\n"
            "2. Odpowiadaj wyłącznie na pytanie – nie dopisuj samodzielnie metadanych ani źródeł.\n"
            "3. Tylko wtedy, gdy podany kontekst w żaden sposób nie wiąże się z pytaniem, napisz wyłącznie:\n"
            "   'Nie jestem w stanie udzielić odpowiedzi na zadane pytanie na podstawie dostępnych dokumentów.'\n"
            "4. Na podziękowania odpowiedz krótko: 'Jeśli masz jeszcze jakieś pytania, chętnie pomogę!'.\n"
            "5. Na powitania odpowiedz krótko: 'Witaj, w czym mogę Ci dzisiaj pomóc?'.\n\n"
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

    def add_document(self, file_path: str, file_name: str):
        chunks = self.loader.load_and_split(file_path, file_name)
        
        valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
        
        if not valid_chunks:
            raise ValueError(
                f"Plik '{file_name}' nie zawiera tekstu"
            )
            
        self.vector_store.add_documents(valid_chunks)

    def delete_document(self, file_name: str):
        self.vector_store.delete_document(file_name)

    def generate_streaming_response(self, message: str, history: List[ChatMessage]) -> Generator[str, None, None]:
        """Główna logika RAG zwracająca strumień tokenów na żywo."""
        lc_history = self._convert_history(history)

        # 1. Query Rewriting
        if lc_history:
            try:
                prompt = self.contextualize_q_prompt.format_messages(
                    chat_history=lc_history, question=message
                )
                standalone_question = self.llm.invoke(prompt).content.strip()
            except Exception as ex:
                print(f"[Query Rewrite Error]: {ex}")
                standalone_question = message
        else:
            standalone_question = message
            

        # 2. Retrieval & Re-ranking
        context_text = ""
        reranked_docs = []

        try:
            initial_docs = self.vector_store.search(standalone_question, k=20)
            
            reranked_docs = self.reranker.rerank(standalone_question, initial_docs, top_n=4)
            
            formatted_chunks = []
            for doc in reranked_docs:
                # 1. Bezpieczne pobranie metadanych
                meta = getattr(doc, "metadata", {}) or {}

                # 2. Nazwa pliku
                source_raw = meta.get("source") or meta.get("file_name") or "dokument.pdf"
                source_name = os.path.basename(str(source_raw))

                # 3. Pobranie numeru strony
                page = meta.get("page")
                page_str = f" (strona {page + 1})" if page is not None else ""

                formatted_chunks.append(f"[Dokument: {source_name}{page_str}]\n{doc.page_content}")
            
            # --- TEJ LINIJKI BRAKOWAŁO: ZŁĄCZENIE LISTY W STRING ---
            context_text = "\n\n---\n\n".join(formatted_chunks)
                
        except Exception as ex:
            print(f"[Retrieval Error]: {ex}")
            context_text = ""
        
        if not context_text:
            yield "Nie jestem w stanie udzielić odpowiedzi na zadane pytanie."
            return

        # 3. Generation (Strumieniowanie)
        formatted_prompt = self.qa_prompt.format_messages(
            context=context_text,
            chat_history=lc_history,
            question=message
        )

        full_response = "" 
        
        for chunk in self.llm.stream(formatted_prompt):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
        
        is_refusal = "Nie jestem w stanie" in full_response
        is_greeting = any(g in full_response.lower() for g in ["pomogę", "w czym mogę", "dzień dobry"])
                
        if not is_refusal and not is_greeting and reranked_docs:
            sources_dict = {}
            for doc in reranked_docs:
                meta = getattr(doc, "metadata", {}) or {}
                source_raw = meta.get("source") or meta.get("file_name") or "dokument.pdf"
                source_name = os.path.basename(str(source_raw))
                page = meta.get("page")

                if source_name not in sources_dict:
                    sources_dict[source_name] = set()
                if page is not None:
                    sources_dict[source_name].add(page + 1)
            
            formatted_sources = []
            for name, pages in sources_dict.items():
                if pages:
                    pages_str = ", ".join(str(p) for p in sorted(pages))
                    formatted_sources.append(f"{name} (strona {pages_str})")
                else:
                    formatted_sources.append(name)

            if formatted_sources:
                sources_output = "\n\nŹródła: " + "; ".join(formatted_sources)
                yield sources_output
                
    def generate_response(self, message: str, history: List[ChatMessage]) -> str:
        return "".join(list(self.generate_streaming_response(message, history)))
        
# Singleton dla całej aplikacji
rag_service = RAGService()