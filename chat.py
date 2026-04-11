from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vector_db import get_embeddings, get_vector_store


PROMPT_TEMPLATE = """Olet kokenut lääketieteen asiantuntija. Tehtäväsi on vastata käyttäjän kysymykseen ammattimaisesti ja selkeästi tietokannasta haetun sisällön perusteella.

Ohjeet:
- Vastaa aina samalla kielellä kuin kysymys on esitetty.
- Käytä täsmällistä, ammattimaista kieltä – vältä epämääräisiä ilmaisuja.
- Rakenna vastaus loogisesti: määrittele ensin ilmiö tai tila, sitten oireet tai syyt, sitten hoito tai toimenpiteet tarpeen mukaan.
- Jos tietokanta sisältää riittävästi tietoa, anna kattava vastaus ilman varauksia.
- Jos tietokanta ei sisällä riittävästi tietoa kysymykseen vastaamiseksi, ilmoita se lyhyesti ja kehota kääntymään lääkärin puoleen.
- Älä koskaan kehota lukijaa etsimään lisätietoa muualta tietokannasta tai muista artikkeleista – tietokanta on jo haettu ja kaikki relevantti tieto on alla olevassa kontekstissa.
- Älä viittaa lähteisiin nimeltä tai mainitse tiedostojen nimiä vastauksessasi.

Konteksti tietokannasta:
{context}

Kysymys: {question}

Vastaus:"""


def build_rag_chain():
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    return retriever, chain


def ask(question: str, retriever, chain) -> tuple[str, list]:
    """Returns (answer, source_docs)."""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs


if __name__ == "__main__":
    print("Loading RAG pipeline...")
    retriever, chain = build_rag_chain()
    print("MDAI – a doctor in your pocket\n(type 'exit' to quit)\n")
    while True:
        question = input("How can I help you? ")
        if question.lower() == "exit":
            break
        answer, docs = ask(question, retriever, chain)
        print(f"\nAnswer:\n{answer}\n")
        print("Sources:", [d.metadata.get("source", "?") for d in docs])
        print("\n" + "=" * 60 + "\n")
