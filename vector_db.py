from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os


def chunk_markdown_files(markdown_dir="terveys_markdown"):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        return_each_line=False,
        strip_headers=True
    )

    documents = []

    for filename in os.listdir(markdown_dir):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(markdown_dir, filename), encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.split_text(text)
        for chunk in chunks:
            chunk.metadata["source"] = filename
        documents.extend(chunks)

    return documents


def batch(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


# Settings
db_location = "./terveys_chroma_db"
collection_name = "terveys_articles"
BATCH_SIZE = 5000

# Embeddings and vector DB setup
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    collection_name=collection_name,
    persist_directory=db_location,
    embedding_function=embeddings,
)

# Load and split markdown files
docs = chunk_markdown_files("terveys_markdown")

# Add in safe batches
for i, chunk in enumerate(batch(docs, BATCH_SIZE)):
    print(f"🔹 Adding batch {i + 1} with {len(chunk)} documents")
    vector_store.add_documents(chunk)
    