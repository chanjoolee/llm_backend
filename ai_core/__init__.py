import os
import app.config

AI_CORE_ROOT_DIR = os.getenv("DAISY_ROOT_FOLDER",os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_DEFAULT_PERSIST_DIR = AI_CORE_ROOT_DIR + "/data/daisy_chromadb"
CHROMA_DB_TEST_DATA_PATH = AI_CORE_ROOT_DIR + "/data/chromadb_test_data.txt"
DATA_SOURCE_SAVE_BASE_DIR = AI_CORE_ROOT_DIR + "/data/data_sources"
