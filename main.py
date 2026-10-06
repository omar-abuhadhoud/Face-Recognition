from config import PERSON_EMBEDDING_METHOD, TEST_IMAGE_PATHS
from image_loading import load_image, load_person_images, list_person_names
from embeddings import (
    get_face_embedding, get_face_embeddings, compute_all_person_embeddings,
    compare_face_to_people, fit_2d_projection, project_to_2d, reduce_groups_to_2d,
)
from plotting import plot_face_against_clusters


def print_similarity_scores(image_label, similarity_by_person):
    """Print {person_name: similarity} sorted from most to least similar."""
    print(f"\nCosine similarity for {image_label}:")
    for person_name, similarity in sorted(similarity_by_person.items(), key=lambda item: item[1], reverse=True):
        print(f"{person_name}: {similarity:.4f}")


if __name__ == "__main__":
    # Build one embedding per person, once
    face_embeddings_by_person = {}
    for person_name in list_person_names():
        _, face_embeddings = get_face_embeddings(load_person_images(person_name))
        face_embeddings_by_person[person_name] = face_embeddings
    person_names, person_embeddings = compute_all_person_embeddings(face_embeddings_by_person, PERSON_EMBEDDING_METHOD)

    # Compare each test image against every person
    test_face_embeddings = {}
    for test_image_path in TEST_IMAGE_PATHS:
        face_embedding = get_face_embedding(load_image(test_image_path))
        if face_embedding is None:
            print(f"No face found in {test_image_path}")
            continue
        test_face_embeddings[test_image_path] = face_embedding
        similarity_by_person = compare_face_to_people(face_embedding, person_names, person_embeddings)
        print_similarity_scores(test_image_path, similarity_by_person)

    # Plot each test face against all the person clusters
    projection = fit_2d_projection(face_embeddings_by_person)
    groups_2d = reduce_groups_to_2d(face_embeddings_by_person, person_names, person_embeddings, projection)
    for test_image_path, face_embedding in test_face_embeddings.items():
        face_point_2d = project_to_2d(face_embedding, projection)
        plot_face_against_clusters(test_image_path, face_point_2d, groups_2d)
