import cv2
img = cv2.imread("images/registered_user.jpg")
if img is None:
    print("Image cannot be opened!")
else:
    print("Image loaded successfully!")
    cv2.imshow("Registered Face", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
