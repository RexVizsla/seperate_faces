from mtcnn import MTCNN
import os
import cv2
import math

# Create the MTCNN detector
detector = MTCNN()

# Set the directory where the images are located
image_dir = r"C:\Users\rexvizsla\Desktop\AI\seperate_faces\input"

# Create a folder for the faces
if not os.path.exists("faces_output"):
    os.mkdir("faces_output")

# Ask if the whole image, just 512x512 or the face only should be saved
while True:
    save_whole_image = input("Do you want to the save the entire image, the face with a 512x512 bounding box around the face or just the face itself? (1/2/3)\n"
                            "1) entire image\n"
                            "2) 512x512 bouding box\n"
                            "3) face only\n")
    if save_whole_image in ["1", "2", "3"]:
        break
    else:
        print("The input was invalid. Please enter either 1, 2 or 3.\n")
 
# Ask if the file name should be kept or changed
while True:
    keep_file_name = input("Do you want to keep the original file name or give them simple ascending numbers? (keep/numbers)\n").lower()
    if keep_file_name in ["keep", "numbers"]:
        break
    else:
        print("Invalid input. Please either type 'keep' to keep the file name of the original image or 'numbers' to set the file name to a simple number.\n")

# Iterate through the images in the directory
for filename in os.listdir(image_dir):
    # Read the image
    image = cv2.imread(os.path.join(image_dir, filename))
    # Detect faces
    faces = detector.detect_faces(image)
    # Iterate through the detected faces
    for face in faces:
        if face['confidence'] > 0.9:
            if save_whole_image == "3":
                x, y, width, height = face['box']
                face_image = image[y:y+height, x:x+width]
            elif save_whole_image == "2":
                print(face)
                centre_x = face['box'][0] + (face['box'][2] // 2)
                centre_y = face['box'][1] - (face['box'][3] // 2)
                corner_x = centre_x - 512//2
                corner_y = centre_y + 512//2
                face_image = image[corner_y:corner_y + 512, corner_x:corner_x + 512]
            elif save_whole_image == "1":
                face_image = image
            if face_image.shape[0]>0 and face_image.shape[1]>0:    
                cv2.imwrite("faces_output/" + filename, face_image)