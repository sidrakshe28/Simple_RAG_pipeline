from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


# 1. Load documents
loader = DirectoryLoader("docs", glob="**/*.txt")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Create Embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create vector store (FAISS)
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Initialize LLM from Ollama
llm = Ollama(model="phi3:mini")


# 6. Create RAG pipeline (Retriever + LLM)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 7. Test query
query = input("Ask something: ")
result = qa_chain.invoke({"query": query})

print("\nAnswer:")
print(result["result"])

print("\nSources:")
for doc in result["source_documents"]:
    print(doc.metadata["source"])
