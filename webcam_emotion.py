import cv2
from deepface import DeepFace
from collections import Counter

cap = cv2.VideoCapture(0)
print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        emotions = []
        for face in results:
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            emotion = face['dominant_emotion']
            emotions.append(emotion)

            # Draw face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Draw emotion label
            cv2.putText(
                frame, emotion, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
            )

        # 📊 Emotion count summary
        counts = Counter(emotions)
        y0 = 30
        for emo, cnt in counts.items():
            cv2.putText(
                frame, f"{emo}: {cnt}", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
            )
            y0 += 30

    except Exception:
        pass

    cv2.imshow("Webcam - Multiple Face Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
