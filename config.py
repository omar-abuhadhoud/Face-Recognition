PEOPLE_DIR = "images"   # contains one folder per person
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
MAX_IMAGE_SIDE = 1920   # larger images are scaled down to this

# Face model
MODEL_NAME = "buffalo_l"                                          # model pack
MODEL_MODULES = ["detection", "recognition"]                      # only what's needed
MODEL_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]   # GPU, CPU as fallback
DEVICE_ID = 0                 # 0 = first GPU, -1 = CPU
DETECTION_THRESHOLD = 0.5     # minimum face detection confidence
DETECTION_SIZE = (640, 640)   # detection input size; larger finds smaller faces
EMBEDDING_SIZE = 512

# PostgreSQL + pgvector (started with docker compose)
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_USER = "face"
POSTGRES_PASSWORD = "face"
POSTGRES_DB = "face_recognition"
DATABASE_URL = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
MATCH_RESULTS = 3   # how many closest people to return per test face
MATCH_FACES = 20    # how many nearest stored faces to fetch before scoring people
MATCH_SCORE_METHOD = "max"   # how to combine a person's nearest faces into one score: "max" or "mean"
MATCH_THRESHOLD = 0.4   # minimum cosine similarity to accept a match; below it the face is "Unknown"

# API server (started with start_server.py)
API_HOST = "0.0.0.0"
API_PORT = 8000
ENROLL_ON_STARTUP = True   # embed PEOPLE_DIR into the database when the server starts
