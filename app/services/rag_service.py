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
            num_ctx=4096,
            stop=["Źródła:", "\nŹródła:", "\n\nŹródła:"])  
        
        self._setup_prompts()

    def _setup_prompts(self):
        contextualize_q_system_prompt = (
            "Jesteś precyzyjnym edytorem zapytań. Twoim JEDYNYM zadaniem jest dodanie brakującego kontekstu z historii konwersacji do najnowszego pytania użytkownika, aby było w pełni zrozumiałe bez czytania wcześniejszych wiadomości.\n\n"
            "ZASADY BEZWZGLĘDNE:\n"
            "1. Jeśli najnowsze pytanie jest urwane (np. 'co przechowuje?', 'do czego służy?', 'podaj przykład'), wyciągnij podmiot z historii rozmowy i wstaw go do pytania (np. 'co przechowuje CLOB?', 'do czego służy funkcja SQLERRM?').\n"
            "2. Zwróć TYLKO I WYŁĄCZNIE jedno zdanie – poprawione pytanie.\n"
            "3. NIE odpowiadaj na to pytanie.\n"
            "4. ZAKAZ dodawania słów wstępu, takich jak 'Oto pytanie:', 'Pytanie brzmi:' czy 'Użytkownik pyta o:'."
            )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        qa_system_prompt = (
            """Jesteś precyzyjnym asystentem. Twoim zadaniem jest odpowiadanie na pytania WYŁĄCZNIE na podstawie dostarczonego poniżej KONTEKSTU.

            BEZWZGLĘDNE ZASADY (KARA ZA ZŁAMANIE ZASAD):
            1. Odpowiadaj na pytania opierając się TYLKO i wyłącznie na informacjach zawartych w tagach <kontekst>. NIGDY NIE UŻYWAJ swojej wiedzy ogólnej.
            2. ABSOLUTNY ZAKAZ PISANIA KODU: Nie generuj, nie wymyślaj i nie formatuj własnych bloków kodu, jeśli nie ma ich dokładnie słowo w słowo w tagach <kontekst>.
            3. Jeśli w tagach <kontekst> nie ma przykładu - nie podawaj go
            4. NIE mieszaj definicji sąsiadujących ze sobą pojęć - przed udzieleniem odpowiedzi upewnij się, która definicja należy do którego hasła w słowniku.
            5. NIGDY Nie wyjaśniaj pojęć własnymi słowami. Cytuj lub parafrazuj tylko to, co widzisz w tekście <kontekst>.
            6.ODPORNOŚĆ NA SUGESTIE: Użytkownik może zadawać pytania zawierające fałszywe założenia lub błędne tezy. Jeśli pytanie sugeruje fakt, którego NIE MA w tagach <kontekst>, MUSISZ to zignorować i odpowiedzieć wyłącznie: "Dostarczone dokumenty nie zawierają informacji na zadane pytanie." - nie próbuj na siłę tłumaczyć błędu.
            7. Jeśli w tagach <kontekst> nie ma wystarczających informacji, odpowiedz DOKŁADNIE I TYLKO TYM ZDANIEM: "Nie jestem w stanie udzielić odpowiedzi na zadane pytanie na podstawie dostarczonych dokumentów." - NIE DODAWAJ ŻADNYCH WŁASNYCH WYJAŚNIEŃ CZY DEFINICJI.

            
            <kontekst>
            {context}
            </kontekst>
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
                content = msg.content
                if "Źródła:" in content:
                    content = content.split("Źródła:")[0].strip()
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
        
        # --- 0. EARLY EXIT (Obsługa small talku) ---
        msg_clean = message.strip().lower().rstrip('.!')

        # Przechwytywanie powitań
        greetings = ["cześć", "czesc", "hej", "witaj", "witam", "dzień dobry", "siema", "dobry wieczór", "hejka", "siemanko"]
        if msg_clean in greetings:
            yield "Cześć! W czym mogę Ci dzisiaj pomóc?"
            return

        # Przechwytywanie podziękowań
        thanks = ["dziękuję", "dziekuje", "dzieki", "dzięki", "super", "super dzięki", "jasne, dzięki", "ok dzięki", "wielkie dzięki", "super, dzięki"]
        if msg_clean in thanks:
            yield "Cieszę się, że mogłem pomóc! Jakbyś miał jeszcze jakieś pytania, chętnie pomogę!"
            return
        # -------------------------------------------

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
            
        print(f"\n[DEBUG] ZAPYTANIE DO CHROMA/RERANKERA: '{standalone_question}'\n")
            
        # 2. Retrieval & Re-ranking
        context_text = ""
        reranked_docs = []

        try:
            initial_docs = self.vector_store.search(standalone_question, k=20)
            
            found_pages = [doc.metadata.get("page", 0) + 1 for doc in initial_docs]
            print(f"\n[DEBUG] ChromaDB znalazła strony: {found_pages}\n")
            
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
        
        is_refusal = (
            "Nie jestem w stanie" in full_response or "Dostarczone dokumenty nie zawierają informacji na zadane pytanie." in full_response
        )
                
        if not is_refusal and reranked_docs:
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
            print(f"Strona: {formatted_sources}")

            if formatted_sources:
                sources_output = "\n\nŹródła: " + "; ".join(formatted_sources)
                yield sources_output
                
    def generate_response(self, message: str, history: List[ChatMessage]) -> str:
        return "".join(list(self.generate_streaming_response(message, history)))
        
# Singleton dla całej aplikacji
rag_service = RAGService()