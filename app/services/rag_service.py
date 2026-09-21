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

class RAGService:
    def __init__(self):
        self.loader = DocumentLoader()
        self.vector_store = VectorStoreManager()
        self.reranker = Reranker()
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.0, num_ctx=4096)
        
        self._setup_prompts()

    def _setup_prompts(self):
        contextualize_q_system_prompt = (
            "Mając daną historię konwersacji oraz najnowsze pytanie użytkownika, "
            "które może odwoływać się do kontekstu w historii, sformułuj samodzielne pytanie, "
            "które można zrozumieć bez znajomości historii konwersacji.\n"
            "ZASADY:\n"
            "- NIE odpowiadaj na pytanie.\n"
            "- Po prostu zmień jego treść jeśli to konieczne, w przeciwnym razie zwróć je bez zmian.\n"
            "- Przekształcone pytanie MUSI być w języku polskim."
            )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        qa_system_prompt = (
            """Jesteś pomocnym asystentem. Odpowiadaj na pytania użytkownika WYŁĄCZNIE na podstawie poniższego kontekstu z dokumentów.
            Rygorystycznie przestrzegaj poniższych zasad:
            1. Odpowiadaj WYŁĄCZNIE na podstawie dostarczonego poniżej KONTEKSTU Z DOKUMENTÓW.
            2. Jeśli w kontekście NIE MA odpowiedzi na pytanie (niezależnie czy pytanie dotyczy wiedzy ogólnej, programowania, Pythona czy czegokolwiek innego), musisz odpowiedzieć DOKŁADNIE i TYLKO tą frazą: "Nie jestem w stanie udzielić odpowiedzi na zadane pytanie."
            3. Zakaz używania własnej wiedzy ogólnej! Jeśli pojęcia (np. 'tuple', 'krotka', 'Python') nie ma w podanym kontekście, traktuj je jako wiedzę nieznaną.
            4. Odpowiedź musi być w języku POLSKIM (wyjątkiem są nazwy własne i fragmenty kodu).
            5. Odpowiedź powinna być zwięzła (maksymalnie 4-5 zdań).
            6. Jeśli użytkownik napisze: dziękuję za pomoc/ do zobaczenia / do widzienia, to wtedy i TYLKO WTEDY odpowiedz 'Jeśli masz jeszcze jakieś pytania to chętnie pomogę!', 
            
            "KONTEKST Z DOKUMENTÓW:\n"
            "{context}\n"
            """
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
        self.vector_store.add_documents(chunks)

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
        try:
            initial_docs = self.vector_store.search(standalone_question, k=20)
            reranked_docs = self.reranker.rerank(standalone_question, initial_docs, top_n=4)
            context_text = "\n\n".join([doc.page_content for doc in reranked_docs])
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

        for chunk in self.llm.stream(formatted_prompt):
            if chunk.content:
                yield chunk.content
    
    def generate_response(self, message: str, history: List[ChatMessage]) -> str:
        return "".join(list(self.generate_streaming_response(message, history)))
        
# Singleton dla całej aplikacji
rag_service = RAGService()