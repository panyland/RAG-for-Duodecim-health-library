from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load embedding function
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
# Koita paraphrase-multilingual-MiniLM-L12-v2 mallia 

# Load vector DB
vector_store = Chroma(
    collection_name="terveys_articles",
    persist_directory="./terveys_chroma_db",
    embedding_function=embeddings,
)

# Make retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Load the LLM
llm = OllamaLLM(model="llama3.2")

# Create the prompt template
template = """
You are a medical expert assistant. Your task is to provide accurate and helpful answers to medical questions based on the provided context. 
The context is in Finnish and may contain medical articles, research papers, or other relevant information. Provide your answers in English.
{context}

Question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

# Chat loop
if __name__ == "__main__":
    print("MDAI - a doctor in your pocket\n(kirjoita 'exit' lopettaaksesi)\n")
    while True:
        question = input("How can I help you? ")
        if question.lower() == "exit":
            break

        # Retrieve docs
        context_docs = retriever.invoke(question)
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        # Generate response
        answer = chain.invoke({"context": context_text, "question": question})
        print("\n🧠 Vastaus:\n")
        print(answer)
        print("\n" + "=" * 40 + "\n")

