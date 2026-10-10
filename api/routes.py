from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import CompareResponse
from embeddings import get_face_embedding
from image_loading import load_image
from recognition import identify

router = APIRouter()


@router.post("/faces/compare", response_model=CompareResponse)
def compare_face(image: UploadFile = File(...)):
    """Compare the face in an uploaded image to the stored faces.
    Returns the matching person's name (null if unknown) and the similarity."""
    face_embedding = get_face_embedding(load_image(image.file))
    if face_embedding is None:
        raise HTTPException(status_code=422, detail="No face found in the image")

    person, similarity = identify(face_embedding)
    if similarity is None:
        raise HTTPException(status_code=404, detail="No faces are stored yet")

    return CompareResponse(person=person, similarity=similarity)
