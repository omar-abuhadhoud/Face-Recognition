from config import MATCH_FACES, MATCH_RESULTS, MATCH_SCORE_METHOD, MATCH_THRESHOLD
from db import save_faces, find_nearest_faces
from embeddings import get_face_embeddings
from image_loading import load_person_images, list_person_names


def enroll_people_from_folder():
    """Embed every image in PEOPLE_DIR and store the faces in the database, one person per folder."""
    image_paths_by_person, face_embeddings_by_person = {}, {}
    for person_name in list_person_names():
        image_paths, face_embeddings = get_face_embeddings(load_person_images(person_name))
        image_paths_by_person[person_name] = image_paths
        face_embeddings_by_person[person_name] = face_embeddings
    save_faces(image_paths_by_person, face_embeddings_by_person)


def find_matches(face_embedding, number_of_results=MATCH_RESULTS, method=MATCH_SCORE_METHOD):
    """Return {person_name: score} for the people closest to face_embedding, most similar first.
    Each person's score combines their nearest stored faces' similarities: "max" or "mean"."""
    similarities_by_person = {}
    for person_name, _, similarity in find_nearest_faces(face_embedding, MATCH_FACES):
        similarities_by_person.setdefault(person_name, []).append(similarity)

    combine = max if method == "max" else (lambda scores: sum(scores) / len(scores))
    score_by_person = {name: combine(scores) for name, scores in similarities_by_person.items()}

    ranked = sorted(score_by_person.items(), key=lambda item: item[1], reverse=True)
    return dict(ranked[:number_of_results])


def identify(face_embedding, threshold=MATCH_THRESHOLD):
    """Return (person_name, similarity) for the best match to face_embedding.
    person_name is None if no one is similar enough; similarity is None if the database is empty."""
    similarity_by_person = find_matches(face_embedding, number_of_results=1)
    if not similarity_by_person:
        return None, None
    best_person, best_similarity = next(iter(similarity_by_person.items()))
    return (best_person if best_similarity >= threshold else None), best_similarity
