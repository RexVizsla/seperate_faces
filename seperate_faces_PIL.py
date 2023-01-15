import cv2
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier(r"C:\Users\rexvizsla\Desktop\AI\seperate_faces\haarcascade_frontalface_default.xml")

# Set the directory where the images are located
image_dir = r"C:\Users\rexvizsla\Desktop\AI\seperate_faces\input"

# Create a folder for the faces
if not os.path.exists("faces_output"):
    os.mkdir("faces_output")

# Iterate through the images in the directory
for filename in os.listdir(image_dir):
    # Read the image
    image = cv2.imread(os.path.join(image_dir, filename))
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # If a face is detected
    if len(faces) > 0:
        # Save the face to the "faces_output" folder
        cv2.imwrite("faces_output/" + filename, image)
