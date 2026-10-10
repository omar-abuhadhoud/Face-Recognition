from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from db.models import Face, Person
from db.session import SessionLocal


def save_faces(image_paths_by_person, face_embeddings_by_person):
    """Take {person_name: image_paths} and {person_name: face_embeddings (N, 512)}.
    Adds any missing people, then inserts or updates one faces row per image path."""
    with SessionLocal.begin() as session:
        for person_name, image_paths in image_paths_by_person.items():
            person_id = session.scalar(
                insert(Person).values(name=person_name)
                .on_conflict_do_update(index_elements=[Person.name], set_={"name": person_name})
                .returning(Person.id)
            )
            rows = [
                {"person_id": person_id, "image_path": image_path, "embedding": embedding}
                for image_path, embedding in zip(image_paths, face_embeddings_by_person[person_name])
            ]
            if not rows:
                continue

            statement = insert(Face).values(rows)
            statement = statement.on_conflict_do_update(
                index_elements=[Face.image_path],
                set_={"person_id": statement.excluded.person_id, "embedding": statement.excluded.embedding},
            )
            session.execute(statement)


def find_nearest_faces(face_embedding, number_of_faces):
    """Return the stored faces closest to face_embedding, most similar first,
    as a list of (person_name, image_path, cosine_similarity)."""
    distance = Face.embedding.cosine_distance(face_embedding)
    query = (
        select(Person.name, Face.image_path, (1 - distance).label("similarity"))
        .select_from(Face)
        .join(Face.person)
        .order_by(distance)
        .limit(number_of_faces)
    )
    with SessionLocal() as session:
        rows = session.execute(query).all()
    return [(name, image_path, float(similarity)) for name, image_path, similarity in rows]
