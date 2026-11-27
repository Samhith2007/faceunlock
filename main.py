import cv2
from deepface import DeepFace
registered_image = "images/registered_user.jpg"
cap = cv2.VideoCapture(0)
print("Camera started!")
print("Click the camera window and press SPACE to verify.")
print("Press Q to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error!")
        break
    cv2.imshow("Face Recognition System", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 32:
        print("Face captured!")
        cv2.imwrite("captured_face.jpg", frame)
        break
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
if key == 32:
    print("Checking face...")
    try:
        result = DeepFace.verify(
            img1_path=registered_image,
            img2_path="captured_face.jpg",
            model_name="Facenet"
        )
        if result["verified"]:
            print("ACCESS GRANTED!")
        else:
            print("ACCESS DENIED!")
    except Exception as e:
        print("Verification error:", e)
