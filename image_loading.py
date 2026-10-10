import os

import cv2
import numpy as np
from PIL import Image, ImageOps

from config import PEOPLE_DIR, IMAGE_EXTENSIONS, MAX_IMAGE_SIDE


def load_image(image_path, max_side=MAX_IMAGE_SIDE):
    """Load a phone photo as a BGR image for InsightFace. Returns None if it fails."""
    try:
        pil_image = Image.open(image_path)
        pil_image = ImageOps.exif_transpose(pil_image)   # fix phone rotation
        pil_image = pil_image.convert("RGB")             # handles RGBA, grayscale, etc.
    except Exception as error:
        print(f"Could not open {image_path}: {error}")
        return None

    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def load_person_images(person_name, max_side=MAX_IMAGE_SIDE):
    """Load every image in PEOPLE_DIR/person_name. Returns a list of (image_path, image),
    where image_path is relative to the program root, e.g. "images/omar/photo.jpg"."""
    person_dir = os.path.join(PEOPLE_DIR, person_name)

    labeled_images = []
    for file_name in sorted(os.listdir(person_dir)):
        if os.path.splitext(file_name)[1].lower() not in IMAGE_EXTENSIONS:
            continue
        image_path = f"{PEOPLE_DIR}/{person_name}/{file_name}"
        image = load_image(image_path, max_side)
        if image is not None:
            labeled_images.append((image_path, image))

    print(f"Loaded {len(labeled_images)} images from {person_dir}")
    return labeled_images


def list_person_names():
    """Return the names of all person folders in PEOPLE_DIR."""
    return [name for name in sorted(os.listdir(PEOPLE_DIR))
            if os.path.isdir(os.path.join(PEOPLE_DIR, name))]
