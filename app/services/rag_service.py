import os
import requests
import time
from typing import List, Generator
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import LLM_MODEL
from app.models import ChatMessage
from app.core.document_loader import DocumentLoader
from app.core.vector_store import VectorStoreManager
from app.core.reranker import Reranker
from app.core.semantic_cache import SemanticCache
class RAGService:
    def __init__(self):
        self.loader = DocumentLoader()
        self.vector_store = VectorStoreManager()
        self.reranker = Reranker()
        
        self.semantic_cache = SemanticCache(threshold=0.2)
        
        self._initialize_llms()
        self._setup_prompts()


    def _initialize_llms(self):
        print("Inicjalizacja modeli LLM...")
        
        # 1. Główny model lokalny (Ollama)
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.ollama_llm = ChatOllama(
            model=LLM_MODEL,
            base_url=self.ollama_url,
            temperature=0.0,
            num_ctx=4096,
            stop=["Źródła:", "\nŹródła:", "Źródło:", "\nŹródło:"]
        ) 
        print(f"Ollama gotowa do pracy na adresie {self.ollama_url}")
        
        # 2. Awaryjny model chmurowy (Gemini)
        self.gemini_llm = None 
        google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if google_api_key:
            print("Znaleziono klucz, inicjalizacja Gemini...")
            try:
                self.gemini_llm = ChatGoogleGenerativeAI(
                    model="gemini-3-flash-preview", 
                    google_api_key=google_api_key,
                    temperature=0.0,
                    max_retries=0,
                    stop=["Źródła:", "\nŹródła:", "Źródło:", "\nŹródło:"]
                )
                print("Gemini gotowe do pracy jako Fallback")
            except Exception as ex:
                print(f"Błąd inicjalizacji Gemini: {ex}")
                self.gemini_llm = None 
        
    def _setup_prompts(self):
        contextualize_q_system_prompt = (
            "Jesteś precyzyjnym edytorem zapytań. Twoim JEDYNYM zadaniem jest przeformułowanie najnowszego pytania użytkownika tak, "
            "aby było w pełni zrozumiałe bez znajomości historii konwersacji.\n\n"
            "ZASADY BEZWZGLĘDNE:\n"
            "1. Wyciągnij z historii TYLKO główny podmiot/temat (np. 'paczka 40kg', 'funkcja w SQL', 'urlop').\n"
            "2. ZAKAZ wklejania faktów, cen, wyników i gotowych odpowiedzi z historii. Masz utworzyć PYTANIE, a nie zdanie twierdzące!\n"
            "3. Zwróć TYLKO I WYŁĄCZNIE samo poprawione pytanie – jedno zdanie.\n"
            "4. ZAKAZ dodawania słów wstępu (takich jak 'Oto pytanie:', 'Użytkownik pyta o:').\n"
            "5. Jeśli zapytanie jest już jasne samo w sobie, zwróć je bez żadnych zmian.\n\n"
            "PRZYKŁADY DO NAŚLADOWANIA:\n"
            "[Historia] Użytkownik: 'Ile kosztuje wersja Premium?' -> AI: 'Kosztuje 20 zł.'\n"
            "[Aktualne pytanie] 'a w opcji rocznej?'\n"
            "[Twój wynik] Ile kosztuje wersja Premium w opcji rocznej?\n\n"
            "[Historia] Użytkownik: 'Co to jest klucz główny w SQL?' -> AI: 'Klucz główny to unikalny identyfikator...'\n"
            "[Aktualne pytanie] 'podaj przykład'\n"
            "[Twój wynik] Podaj przykład klucza głównego w SQL.\n\n"
            "[Historia] Użytkownik: 'Dla paczki 40kg opłata to 44,99 zł.'\n"
            "[Aktualne pytanie] 'a co jeśli wyślę w niedzielę?'\n"
            "[Twój wynik] Jaka jest cena wysłania paczki 40kg w niedzielę?"
            )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Pytanie użytkownika: {question}\n\nZADANIE: Napisz TYLKO I WYŁĄCZNIE przeformułowane zapytanie do wyszukiwarki. ZAKAZ ODPOWIADANIA NA PYTANIE!"),
        ])

        qa_system_prompt = (
            """Jesteś precyzyjnym asystentem. Twoim zadaniem jest odpowiadanie na pytania użytkownika WYŁĄCZNIE na podstawie dostarczonego w tagach <kontekst> tekstu.

            ZASADA ABSOLUTNA: Przeanalizuj pytanie i wybierz TYLKO JEDNĄ z dwóch poniższych ścieżek. Nigdy nie łącz ich ze sobą.

            ŚCIEŻKA A (Odpowiedź znaleziona): 
            Jeśli kontekst zawiera informacje pozwalające na odpowiedź, sformułuj ją opierając się tylko na tym tekście. 
            - Nie używaj wiedzy ogólnej i nie dodawaj własnych definicji.
            - Nie generuj bloków kodu, jeśli nie ma ich w tekście.
            - Po udzieleniu odpowiedzi natychmiast zakończ wypowiedź.

            ŚCIEŻKA B (Brak informacji / Błędne założenia): 
            Jeżeli w kontekście brakuje wystarczających informacji do udzielenia odpowiedzi LUB pytanie sugeruje fakty, których nie ma w tekście, zignoruj pytanie i napisz DOKŁADNIE to jedno zdanie, bez żadnych dodatkowych słów:
            "Nie jestem w stanie udzielić odpowiedzi na zadane pytanie na podstawie dostarczonych dokumentów."

            <kontekst>
            {context}
            </kontekst>"""
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
        
        self.semantic_cache.clear_cache() # wywoływane po dodaniu dokumentu

    def delete_document(self, file_name: str):
        self.vector_store.delete_document(file_name)
        
        self.semantic_cache.clear_cache()

    def generate_streaming_response(self, message: str, history: List[ChatMessage]) -> Generator[str, None, None]:
        """Główna logika RAG zwracająca strumień tokenów na żywo."""
        
        is_error = False 
        # --- 0. EARLY EXIT (Obsługa small talku) ---
        msg_clean = message.strip().lower().rstrip('.!')

        # Przechwytywanie powitań
        greetings = {"cześć", "czesc", "hej", "witaj", "witam", "dzień dobry", "siema", "dobry wieczór", "hejka", "siemanko"}
        if msg_clean in greetings:
            yield "Cześć! W czym mogę Ci dzisiaj pomóc?"
            return

        # Przechwytywanie podziękowań
        thanks = {"dziękuję", "dziekuje", "dzieki", "dzięki", "super", "super dzięki", "jasne, dzięki", "ok dzięki", "wielkie dzięki", "super, dzięki"}
        if msg_clean in thanks:
            yield "Cieszę się, że mogłem pomóc! Jakbyś miał jeszcze jakieś pytania, chętnie pomogę!"
            return
        
        cached_answer = self.semantic_cache.search(msg_clean)
        if cached_answer:
            print("Zwracam odpowiedź z cache'a", flush=True)
            words = cached_answer.split(" ")
            for word in words:
                yield word + " "
                time.sleep(0.02)
            return 
        
        lc_history = self._convert_history(history)

        # 1. Query Rewriting
        if lc_history:
            try:
                prompt = self.contextualize_q_prompt.format_messages(
                    chat_history=lc_history, question=message
                )
                
                try:
                    requests.get(self.ollama_url, timeout=1.5)
                    
                    response_content = self.ollama_llm.invoke(prompt).content
                except Exception as ex:
                    print(f"Ollama padła podczas modyfikacji pytanie {ex}. Uruchamian Gemini")
                    if self.gemini_llm:
                        response_content = self.gemini_llm.invoke(prompt).content
                    else:
                        raise Exception("Brak zapasowego modelu chmurowego")
                
                # Normalizacja: Gemini (lista) vs Ollama (string)
                if isinstance(response_content, list):
                    standalone_question = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item) 
                        for item in response_content
                    ).strip()
                else:
                    standalone_question = str(response_content).strip()
                    
            except Exception as ex:
                print(f"[Query Rewrite Error]: {ex}")
                standalone_question = message
        else:
            standalone_question = message
            
        # 2. Retrieval & Re-ranking
        context_text = ""
        reranked_docs = []

        try:
            
            initial_docs = self.vector_store.search(standalone_question, k=15)
            
            found_pages = [doc.metadata.get("page", 0) + 1 for doc in initial_docs]
            print(f"\n[DEBUG] ChromaDB znalazła strony: {found_pages}\n")
            
            reranked_docs = self.reranker.rerank(standalone_question, initial_docs, top_n=4)
            
            formatted_chunks = []
            for i, doc in enumerate(reranked_docs):
               formatted_chunks.append(f"[Fragment wiedzy nr {i+1}]\n{doc.page_content}")
            
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
        
        def stream_with_model(model):
            nonlocal full_response           
        
            for chunk in model.stream(formatted_prompt):
                chunk_data = chunk.content

                # Gemini -> lista / Ollama -> string
                if isinstance(chunk_data, list):
                    # wyciąganie tekstu ze słowników
                    text_chunk = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item) 
                        for item in chunk_data
                    )
                else:
                    # String dla Ollamy
                    text_chunk = str(chunk_data) if chunk_data else ""

                if text_chunk:
                    full_response += text_chunk
                    yield text_chunk
        
        
        try:
            requests.get(self.ollama_url, timeout=1.5)
            print("\n[INFO] Start strumieniowania z lokalnego Qwena...", flush=True)
            yield from stream_with_model(self.ollama_llm)
            
        except Exception as ex:
            print(f"\n[WARNING] Padła Ollama. Fallback do Gemini...", flush=True)
            try:
                if self.gemini_llm:
                    yield from stream_with_model(self.gemini_llm)
                else:
                    raise Exception("Model gemini nie zainicjalizowany")
            except Exception as ex2:
                yield "\nSystemy AI są obecnie niedostępne. Spróbuj ponownie później."
                is_error = True
                
        is_refusal = (
            "Nie jestem w stanie" in full_response or "Dostarczone dokumenty nie zawierają informacji na zadane pytanie." in full_response
        )
                
        if not is_refusal and reranked_docs and not is_error:
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
                full_response += sources_output
                if not is_error and not is_refusal:
                    self.semantic_cache.save(msg_clean, full_response)
                    print(f"Zapisano do cache {msg_clean}", flush=True)
                yield sources_output
            else:
                # gdyby nie było źródeł z jakiejś przyczyny
                if not is_error and not is_refusal:
                    self.semantic_cache.save(msg_clean, full_response)
                    print("Zapisano do cache bez źródeł", flush=True)

    def generate_response(self, message: str, history: List[ChatMessage]) -> str:
        return "".join(list(self.generate_streaming_response(message, history)))
        
# Singleton dla całej aplikacji
rag_service = RAGService()