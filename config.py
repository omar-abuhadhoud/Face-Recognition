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

# How to combine one person's face embeddings: "mean" or "median"
PERSON_EMBEDDING_METHOD = "mean"

TEST_IMAGE_PATHS = [
    "test/image.png",
    "test/photo_1_2026-10-06_23-43-14.jpg",
    "test/photo_2_2026-10-06_23-43-14.jpg",
]
