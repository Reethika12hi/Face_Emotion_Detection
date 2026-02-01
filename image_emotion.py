import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import os

# Image path
image_path = r"C:\Users\sudhe\OneDrive\Desktop\Face_Emotion_Detection\women.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError("Image not found")

# Read image
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Analyze all faces
results = DeepFace.analyze(
    img_path=image_path,
    actions=['emotion'],
    enforce_detection=False
)

# Draw bounding box & emotion for each face
for face in results:
    region = face['region']
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    emotion = face['dominant_emotion']

    # Draw rectangle
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Put emotion text
    cv2.putText(
        img_rgb,
        emotion,
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2
    )

# Show result
plt.figure(figsize=(8, 8))
plt.imshow(img_rgb)
plt.axis('off')
plt.title("Multiple Face Emotion Detection")
plt.show()
