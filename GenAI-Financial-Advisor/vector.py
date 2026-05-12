from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursivecharacterTextsplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import chroma
import os
from dotenv import load_dotenv
from collections import OrderedDict

# Load environment variables from .env file
load_dotenv()

os.environ["HUGGINGFACEHUB API TOKEN"] = os.getenv("HUGGINGFACEHUB API TOKEN")