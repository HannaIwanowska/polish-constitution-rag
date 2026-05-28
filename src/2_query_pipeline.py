import os
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings 
from dotenv import load_dotenv

load_dotenv()

KATALOG_CHROMA = "./chroma_db"

PROMPT_SYSTEMOWY = """Jestes ekspertem od Konstytucji RP.
Odpowiadaj WYLACZNIE na podstawie podanego kontekstu.
Zawsze podawaj numer artykulu.
Jesli nie znajdziesz odpowiedzi — powiedz
"Nie znalazlem odpowiedzi w Konstytucji."

KONTEKST:
{context}

PYTANIE: {question}
"""

def format_docs(docs):
    return "\n".join(f"[Art. {doc.metadata.get('artykul', 'Brak')}]: {doc.page_content}" for doc in docs)

def zaladuj_system():
    print("Ładowanie embeddingów i bazy ChromaDB...")
    embeddingi = OpenAIEmbeddings(model="text-embedding-3-small")
    baza_wiedzy = Chroma(persist_directory=KATALOG_CHROMA, embedding_function=embeddingi)

    print("Ładowanie modelu OpenAI (gpt-4o)...")
    llm = ChatOpenAI(
        model="gpt-4o",      
        temperature=0.0      
    )
    return baza_wiedzy, llm

if __name__ == "__main__":
    baza, llm = zaladuj_system()

    retriever = baza.as_retriever(search_kwargs={"k": 15}) 
    prompt = PromptTemplate.from_template(PROMPT_SYSTEMOWY)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n✅ System RAG oparty na OpenAI gotowy do działania!\n")

pytania = [
    "Kto powołuje Prezesa Rady Ministrów?", 
    "Ile lat trwa kadencja Sejmu i Senatu?",
    "Kto może być Prezesem Rady Ministrów?", 
    "Z ilu posłów składa się Sejm Rzeczypospolitej Polskiej?", 
    "Kto w Rzeczypospolitej Polskiej stosuje prawo łaski?", 
    "Jak upiec idealną szarlotkę?"
]

for pyt in pytania:
    print(f"\n--- PYTANIE: {pyt} ---")
    odpowiedz = rag_chain.invoke(pyt)
    print(f"ODPOWIEDŹ:\n{odpowiedz.strip()}\n")
    print("-" * 50)