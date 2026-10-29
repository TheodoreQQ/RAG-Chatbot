import os
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
class RAGService:
    def __init__(self):
        self.loader = DocumentLoader()
        self.vector_store = VectorStoreManager()
        self.reranker = Reranker()
        
        self._initialize_llms()
        self._setup_prompts()


    def _initialize_llms(self):
        print("Uruchamiam główny model Ollama Qwen2.5")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.ollama_llm = ChatOllama(
            model=LLM_MODEL,
            base_url=ollama_url,
            temperature=0.0,
            num_ctx=4096,
            stop=["Źródła:", "\nŹródła:", "\n\nŹródła:", "Źródło:", "\nŹródło:", "\n\nŹródło:"]
        ) 
        self.gemini_llm = None 
        google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if google_api_key:
            print("Znaleziono klucz, próba inicjalizacji Gemini")
            try:
                gemini_llm = ChatGoogleGenerativeAI(
                    model="gemini-3-flash-preview",
                    google_api_key=google_api_key,
                    temperature=0.0,
                    max_retries=0,
                    stop=["Źródła:", "\nŹródła:", "\n\nŹródła:", "Źródło:", "\nŹródło:", "\n\nŹródło:" ]
                )
                
                print("Gemini ready to work")
        
            except Exception as ex:
                print(f"Something went wrong {ex}")
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
        
        lc_history = self._convert_history(history)

        # 1. Query Rewriting
        if lc_history:
            try:
                prompt = self.contextualize_q_prompt.format_messages(
                    chat_history=lc_history, question=message
                )
                
                try:
                    response_content = self.ollama_llm.invoke(prompt).content
                except Exception as ex:
                    print(f"[WARNING] Ollama padła podczas modyfikacji pytanie {ex}. Uruchamian Gemini")
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
            nonlocal full_response, first_token            
        
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
            print("\n[INFO] Start strumieniowania z lokalnego Qwena...", flush=True)
            yield from stream_with_model(self.ollama_llm)
        except Exception as ex:
            print(f"\n[WARNING] Błąd lokalnego modelu Ollama: {ex}", flush=True)
            print("[INFO] Odpalam koło ratunkowe: przełączam do chmury Gemini...", flush=True)   
            
            try:
                if self.gemini_llm:
                    first_token = True 
                    yield from stream_with_model(self.gemini_llm)
                else:
                    raise Exception("Model gemini nie został zainicjalizowany")
            except Exception as ex:
                print("Gemini zawiodło", flush=True)
                yield "\n Systemy AI są obecnie niedostępne. Spróbuj ponownie później"
                
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