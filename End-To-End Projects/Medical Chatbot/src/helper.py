from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# Extract Data From the PDF File
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    
    documents = loader.load()
    
    return documents



# Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )

    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks



# Initialize the Embeddings
def init_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings