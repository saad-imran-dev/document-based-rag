import os
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document


# PDF Doc
def load_pdf_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs


def split_documents(docs: list[Document], chunk_size=1500, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


# Vector Store
def init_vectorstore():
    # Initialize the vector store with Google Generative AI embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma")
    return vectorstore


def retriever(vectorstore: Chroma, query: str):
    results = vectorstore.similarity_search(query, k=5)
    resultMessages = [
        SystemMessage(content=f"Document {i+1}: {doc.page_content}")
        for i, doc in enumerate(results)
    ]
    return resultMessages


def load_docs_to_vectorstore(pdf_dir_path: str, vectorstore: Chroma):
    if not os.path.exists(pdf_dir_path):
        print(f"Directory {pdf_dir_path} does not exist.")
        raise FileNotFoundError(f"Directory {pdf_dir_path} does not exist.")

    pdf_files = [file for file in os.listdir(pdf_dir_path) if file.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir_path, pdf_file)
        docs = load_pdf_text(pdf_path)
        chunks = split_documents(docs)
        vectorstore.add_documents(chunks)
    print(f"Loaded {len(pdf_files)} PDF files into the vector store.")


# Langchain
def get_chat_model():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def create_chatbot_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers questions regarding company policy based on provided policy document excerpts.",
            ),
            (
                "system",
                "Use only the relevant information from the retrieved documents to answer the question. "
                "Do not quote the documents directly; instead, synthesize a clear and complete answer.",
            ),
            MessagesPlaceholder(variable_name="documents"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}"),
        ]
    )
    llm = get_chat_model()
    chain = prompt | llm
    return chain


def create_query_rewrite_chain():
    # Rewrite user queries with history context for fetching relevant documents
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a query rewriting assistant. Given a current user query and a list of previous queries (chat history), "
                "rewrite the current query to make it fully self-contained by incorporating relevant context from the previous queries. "
                "Only include information from the history if it is necessary for understanding or answering the current query.",
            ),
            ("human", "Chat History:\n{history}"),
            ("human", "Current query: {query}"),
        ]
    )
    llm = get_chat_model()
    chain = prompt | llm
    return chain


def refine_query(query: str, history: list):
    # Use the query rewrite chain to refine the user query with history context
    if len(history) == 0:
        return query

    user_query_history = "\n".join(
        [message.content for message in history if isinstance(message, HumanMessage)]
    )
    rewrite_chain = create_query_rewrite_chain()
    rewritten_query = rewrite_chain.invoke(
        {"history": user_query_history, "query": query}
    )
    print(f"Rewritten Query: {rewritten_query.content}")
    return rewritten_query.content


if __name__ == "__main__":
    dotenv.load_dotenv()
    pdf_dir_path = os.path.abspath(os.path.join(os.curdir, os.getenv("PDF_DIRECTORY")))
    vectorstore = init_vectorstore()
    load_docs_to_vectorstore(pdf_dir_path, vectorstore)
    chatbot = create_chatbot_chain()

    user_query = input("Enter your query: ")
    history = []
    while user_query.lower() != "exit":
        query_with_context = refine_query(user_query, history)
        retriever_results = retriever(vectorstore, query_with_context)
        response = chatbot.invoke(
            {"documents": retriever_results, "query": user_query, "history": history}
        )
        print(f"AI: {response.content}\n\n")
        history.extend([HumanMessage(content=user_query), response])
        user_query = input("Enter your query (or type 'exit' to quit): ")
