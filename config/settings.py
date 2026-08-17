import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH="storage/chroma"
COLLECTION_NAME="catbot_knowledge"
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=5