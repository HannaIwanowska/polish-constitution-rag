import re
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings # NOWY IMPORT
from dotenv import load_dotenv

load_dotenv()

# --- 1. FUNKCJA DO CHUNKINGU ---
def przygotuj_chunki(sciezka_do_pliku):
    with open(sciezka_do_pliku, 'r', encoding='utf-8') as f:
        tekst = f.read()

    dokumenty = []
    aktualny_rozdzial = "Preambuła"

    # Usuwamy z tekstu ``` żeby nie brudziły nam bazy
    tekst_czysty = tekst.replace("```", "")

    # Dzielimy cały tekst po słowie "Art. " oraz używamy regexa, żeby zachować słowo "Art. " w wynikach
    fragmenty = re.split(r'(?=Art\. \d+)', tekst_czysty)

    for fragment in fragmenty:
        fragment = fragment.strip()
        if not fragment:
            continue
            
        # Szukamy, czy w tym fragmencie zmienił się rozdział
        rozdzial_szukaj = re.search(r'Rozdział\s+([IVXLCDM]+)', fragment)
        if rozdzial_szukaj:
            aktualny_rozdzial = rozdzial_szukaj.group(1)
            
        # Szukamy numeru artykułu do metadanych
        artykul_szukaj = re.search(r'Art\.\s+(\d+[a-z]*)\.', fragment)
        
        if artykul_szukaj:
            numer_artykulu = artykul_szukaj.group(1)
            # Tworzymy obiekt Document, który LangChain rozumie
            doc = Document(
                page_content=fragment,
                metadata={"rozdzial": aktualny_rozdzial, "artykul": numer_artykulu}
            )
            dokumenty.append(doc)
        else:
            # Jeśli nie ma "Art.", to pewnie Preambuła
            if "W trosce o byt" in fragment:
                doc = Document(
                    page_content=fragment,
                    metadata={"rozdzial": "Preambuła", "artykul": "Preambuła"}
                )
                dokumenty.append(doc)

    print(f"✅ Znaleziono i pocięto {len(dokumenty)} artykułów.")
    return dokumenty

# --- 2. GŁÓWNY PIPELINE INDEKSOWANIA ---
def stworz_baze_wektorowa():
    sciezka_dane = "data/konstytucja.md"
    katalog_chroma = "./chroma_db"

    print("Ładowanie i cięcie tekstu Konstytucji...")
    dokumenty = przygotuj_chunki(sciezka_dane)

    print("Pobieranie modelu do embeddingów (OpenAI)...")
    # NOWY MODEL EMBEDDINGOWY
    model_embeddingow = OpenAIEmbeddings(model="text-embedding-3-small") 

    print("Tworzenie bazy wektorowej ChromaDB...")
    baza = Chroma.from_documents(
        documents=dokumenty,
        embedding=model_embeddingow,
        persist_directory=katalog_chroma
    )
    print("✅ Baza wektorowa gotowa!")

if __name__ == "__main__":
    # Upewnij się, że masz stworzony folder data i wrzucony do niego plik!
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Stworzono folder 'data'. Wrzuć tam plik 'konstytucja.md' i uruchom ponownie.")
    else:
        stworz_baze_wektorowa()



