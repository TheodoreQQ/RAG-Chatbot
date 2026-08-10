import os 
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Pobranie pliku i wektoryzacja
file_path = os.getenv('FILEPATH_BW')

loader = PyPDFLoader(file_path)
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000, 
  chunk_overlap=150
  )
chunks = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")


# tworzenie bazy wektorowej i zapisanie jej w folderze ./chroma_db
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Szukamy top 2 najbardziej pasujących fragmentów
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# inicjalizacja modelu oraz promptu
llm = ChatOllama(
  model="qwen2.5:7b",
  temperature=0.0,
  num_ctx=4096,
  keep_alive="30m"
)

prompt_template = ChatPromptTemplate.from_template(
 """Jesteś pomocnym asystentem i ekspertem. 
Odpowiadaj na pytanie użytkownika na podstawie podanego KONTEKSTU. Staraj się odpowiadać maksymalnie w kilku zdaniach.

ZASADY:
1. Kontekst może zawierać pojęcia, nazwy funkcji, kodu lub nagłówki w języku angielskim. Przeanalizuj je i powiąż z pytaniem użytkownika.
2. Jeśli pytanie dotyczy funkcji lub pojęcia z kontekstu, podaj jej dokładną nazwę oraz wyjaśnienie.
3. Jeśli w KONTEKŚCIE absolutnie nie ma informacji na ten temat, napisz: "Nie jestem w stanie znaleźć informacji w dostarczonych dokumentach" Nie pisz nic więcej wtedy, nie zadawaj pytań pomocnicznych ani nic w tym rodzaju.

KONTEKST Z DOKUMENTÓW:
{context}

PYTANIE UŻYTKOWNIKA:
{question}
"""
)

print("=== System RAG, Zadaj pytanie na podstawie pliku PDF ===")
print("--- Wpisz exit aby zakończyć rozmowę ---")

# Pętla czatu

while(True):
  
  user_input = input("Ty: ").strip()
  
  if user_input.lower() in ['exit', 'quit', 'wyjście']:
    print("Asystent: Siema")
    break
  
  if not user_input:
    continue
  
  # Wyszukiwanie relewantnych dokumentów
  relevant_docs = retriever.invoke(user_input)
  
  # Łączymy ich treść w jeden ciąg tekstu
  context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
  
  # Formatowanie prompta przekazując kontekst oraz pytanie
  formatted_prompt = prompt_template.format(context=context_text, question=user_input)
  
  # Odpowiedź ze streamingiem 
  print("Asystent: ", end="", flush=True)
  try: 
    for chunk in llm.stream(formatted_prompt):
      print(chunk.content, end="", flush=True)
    print("\n")
  except Exception as e:
    print(f"\n błąd połączenia z Ollamą: {e}")