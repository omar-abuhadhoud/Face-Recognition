import numpy as np
from insightface.app import FaceAnalysis

from config import (
    MODEL_NAME, MODEL_MODULES, MODEL_PROVIDERS, DEVICE_ID,
    DETECTION_THRESHOLD, DETECTION_SIZE, EMBEDDING_SIZE,
)

face_model = FaceAnalysis(name=MODEL_NAME, allowed_modules=MODEL_MODULES, providers=MODEL_PROVIDERS)
face_model.prepare(ctx_id=DEVICE_ID, det_thresh=DETECTION_THRESHOLD, det_size=DETECTION_SIZE)


def get_face_embedding(image):
    """Take an image from load_image() and return its face embedding (512 numbers).
    Returns None if no face is found."""
    if image is None:
        return None

    faces = face_model.get(image, max_num=1)   # most prominent face only
    if not faces:
        return None

    return faces[0].normed_embedding


def get_face_embeddings(labeled_images):
    """Take a list of (label, image) and return (labels, face_embeddings).
    face_embeddings is a NumPy array of shape (N, 512). Images with no face are skipped."""
    labels, face_embeddings = [], []

    for label, image in labeled_images:
        face_embedding = get_face_embedding(image)
        if face_embedding is None:
            print(f"No face found in {label}")
            continue
        labels.append(label)
        face_embeddings.append(face_embedding)

    if not face_embeddings:
        return [], np.empty((0, EMBEDDING_SIZE), dtype=np.float32)

    return labels, np.stack(face_embeddings).astype(np.float32)


def cosine_similarity(embedding_a, embedding_b):
    """Return the cosine similarity between two normalized embeddings."""
    return float(np.dot(embedding_a, embedding_b))   # normed embeddings, so dot = cosine similarity


def compute_person_embedding(face_embeddings, method="mean"):
    """Take one person's face embeddings (N, 512) and combine them into one normalized embedding.
    method: "mean" or "median". Returns None if face_embeddings is empty."""
    if len(face_embeddings) == 0:
        return None

    if method == "median":
        person_embedding = np.median(face_embeddings, axis=0)
    else:
        person_embedding = np.mean(face_embeddings, axis=0)

    return person_embedding / np.linalg.norm(person_embedding)   # re-normalize so dot = cosine similarity


def compute_all_person_embeddings(face_embeddings_by_person, method="mean"):
    """Take {person_name: face_embeddings} and build one embedding per person.
    Returns (person_names, person_embeddings) where person_embeddings has shape (number_of_people, 512)."""
    person_names, person_embeddings = [], []
    for person_name, face_embeddings in face_embeddings_by_person.items():
        person_embedding = compute_person_embedding(face_embeddings, method)
        if person_embedding is None:
            print(f"No faces found for {person_name}")
            continue
        person_names.append(person_name)
        person_embeddings.append(person_embedding)

    if not person_embeddings:
        return [], np.empty((0, EMBEDDING_SIZE), dtype=np.float32)
    return person_names, np.stack(person_embeddings).astype(np.float32)


def fit_2d_projection(face_embeddings_by_person):
    """Learn a PCA projection from 512D to 2D using every person's face embeddings.
    Returns (mean, top_2_components) to pass to project_to_2d()."""
    all_face_embeddings = np.vstack(list(face_embeddings_by_person.values()))
    mean = all_face_embeddings.mean(axis=0)
    _, _, components = np.linalg.svd(all_face_embeddings - mean, full_matrices=False)
    top_2_components = components[:2].T   # (512, 2)
    return mean, top_2_components


def project_to_2d(embeddings, projection):
    """Project embeddings (N, 512) or one embedding (512,) to 2D with a projection from fit_2d_projection()."""
    mean, top_2_components = projection
    return (embeddings - mean) @ top_2_components


def reduce_groups_to_2d(face_embeddings_by_person, person_names, person_embeddings, projection):
    """Project every person's face embeddings and person embedding to 2D with the same projection.
    Returns {person_name: (face_points_2d (N, 2), person_point_2d (2,))}."""
    groups_2d = {}
    for person_name, person_embedding in zip(person_names, person_embeddings):
        face_points_2d = project_to_2d(face_embeddings_by_person[person_name], projection)
        person_point_2d = project_to_2d(person_embedding, projection)
        groups_2d[person_name] = (face_points_2d, person_point_2d)
    return groups_2d


def compare_face_to_people(face_embedding, person_names, person_embeddings):
    """Take one face embedding and every person's embedding.
    Returns {person_name: cosine similarity}."""
    similarity_scores = person_embeddings @ face_embedding   # one dot product per person
    return {name: float(score) for name, score in zip(person_names, similarity_scores)}
