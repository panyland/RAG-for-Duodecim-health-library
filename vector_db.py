from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os


EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DB_LOCATION = "./terveys_chroma_db"
COLLECTION_NAME = "terveys_articles"
BATCH_SIZE = 5000


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def get_vector_store(embeddings=None):
    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings,
    )


def chunk_markdown_files(markdown_dir="terveys_markdown"):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        return_each_line=False,
        strip_headers=True,
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
        yield iterable[i : i + batch_size]


if __name__ == "__main__":
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    print("Chunking markdown files...")
    docs = chunk_markdown_files("terveys_markdown")
    print(f"Total chunks: {len(docs)}")

    for i, chunk in enumerate(batch(docs, BATCH_SIZE)):
        print(f"Adding batch {i + 1} ({len(chunk)} documents)...")
        vector_store.add_documents(chunk)

    print("Done.")
