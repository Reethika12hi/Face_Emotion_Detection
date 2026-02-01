Face Emotion Detection Project – Description
 Project Title: Face Emotion Detection using Deep Learning and OpenCV
 
This project implements a real-time Face Emotion Detection system using Deep Learning and Computer Vision techniques. The application can analyze facial expressions from both static images and live webcam video, identify multiple faces simultaneously, and classify emotions such as happy, sad, angry, surprise, fear, disgust, and neutral.

The system uses OpenCV for image and video processing, and DeepFace, a pre-trained deep learning framework, for accurate emotion recognition. In webcam mode, the application displays bounding boxes around detected faces, labels each face with its predicted emotion, and shows a live emotion count summary on the screen.

This project demonstrates practical knowledge of Python, OpenCV, Deep Learning, model integration, and real-time processing, making it suitable for applications in human–computer interaction, mental health monitoring, and user behavior analysis.

Technologies Used:-

Python

OpenCV

DeepFace

TensorFlow

tf-keras

 Key Features

Detects multiple faces in a single image

Predicts emotion for each detected face

Supports real-time webcam emotion detection

Displays emotion labels and bounding boxes

Shows live emotion count summary 

Keyboard controls for quitting and screenshots
<img width="800" height="800" alt="Detect face emotions" src="https://github.com/user-attachments/assets/f0ed05fe-0ac2-44bb-92a9-402fcc542893" />
<img width="800" height="800" alt="Detect face emotions" src="https://github.com/user-attachments/assets/3817d165-8bbe-4d19-8e39-ca249b1218d9" />

How to Run
pip install -r requirements.txt
python image_emotion.py      # Image-based detection
python webcam_emotion.py     # Real-time webcam detection

Learning Outcomes:

Understanding of facial emotion recognition

Hands-on experience with pre-trained deep learning models

Real-time video processing using OpenCV

Practical implementation of computer vision pipelines
