from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


Settings.embed_model = GoogleGenAIEmbedding(model="models/embedding-001")

data = SimpleDirectoryReader('articles')

documents = data.load_data()

index = VectorStoreIndex.from_documents(documents)

index.storage_context.persist()