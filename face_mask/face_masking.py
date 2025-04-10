import cv2

# Load pre-trained classifier for face detection
face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cas.empty():
    raise IOError("Failed to load face cascade classifier.")

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to read from video capture device.")
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_roi = img[y:y + h, x:x + w]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
        img[y:y + h, x:x + w] = blurred_face
        
    cv2.imshow("Face Blurring", img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#close the display window
cap.release()
cv2.destroyAllWindows()

#prepare by kishan rajoria