from deepface import DeepFace
image = "images/registered_user.jpg"
try:
    faces = DeepFace.extract_faces(
        img_path=image,
        detector_backend="retinaface"
    )
    print("Face detected successfully!")
    print("Number of faces:", len(faces))
except Exception as e:
    print("Face detection failed!")
    print("Error:", e)
