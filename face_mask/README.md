# Face Masking with OpenCV

This project demonstrates a simple face masking (blurring) application using OpenCV. The application uses a pre-trained Haar Cascade Classifier to detect faces in real-time from a webcam feed and applies a Gaussian blur to obscure the detected faces.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x installed on your machine
- OpenCV library installed (`opencv-python`)
- A webcam connected to your machine

## Installation

1. **Clone the repository**:

   ```sh
   git clone https://github.com/kishan-rajoria/VisionGrid.git
   cd VisionGrid/face_masking
   ```

2. **Install the required packages**:

   ```sh
   pip install opencv-python
   ```

## Usage

To run the face masking script, execute the following command:

```sh
python face_masking.py
```

The script will start capturing video from your webcam, detect faces, and apply a Gaussian blur to them in real-time.

### Key Functionality

- **Face Detection**: Uses OpenCV's Haar Cascade Classifier to detect faces in the video stream.
- **Face Blurring**: Applies a Gaussian blur to the detected faces.

### Example Code

Below is the complete example code used in the project:

```python
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

# Close the display window
cap.release()
cv2.destroyAllWindows()
```

### How It Works

1. **Initialize the Cascade Classifier**: Loads the pre-trained face detection model.
2. **Capture Video**: Starts capturing video from the webcam.
3. **Face Detection**: Converts each frame to grayscale and detects faces.
4. **Face Blurring**: Applies a Gaussian blur to each detected face region.
5. **Display Output**: Shows the processed video in a window.
6. **Exit on 'q' key press**: Allows the user to exit the application by pressing the 'q' key.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
