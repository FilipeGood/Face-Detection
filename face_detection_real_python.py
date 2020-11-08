"""
    Detects faces in a given photo
    python3 face_detection_real_python.py  faces2.jpg  haarcascade_frontalface_default.xml
"""

# url => https://realpython.com/face-recognition-with-python/

import cv2
import sys


# Get images
imagePath = sys.argv[1]
casc_path = "haarcascade_frontalface_default.xml" # XML file that contains the data to detect faces


print(imagePath)
print(casc_path)


# Create the haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")




# Read the image and covert it to grayscale
image = cv2.imread(imagePath)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Detect faces in the image
"""
    1. The detectMultiScalse function is a general function that detects objects. Since we are
        calling it on the face cascade, that's what it detects
    2. scaleFactor => since some faces may be closer to the camera, they would appear bigger than
        the faces in the back. The scale factor compensates for this
    3. The detection algorithm uses a moving window to detect objects.
        minNeighbors defines how many objects are detected near the current one before
        it declares the face found. minSize, meanwhile, gives the size of each window.

    The function returns a list of rectangles in which it believes it found a face.
    Returns for values.
        - x and y => location of the rectangle
        - w and h => width and height of the rectangle
"""
faces = face_cascade.detectMultiScale(
    gray_img,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)


print("Fount {0} faces!".format(len(faces)))

print(faces)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


cv2.imshow("Faces found", image)
cv2.waitKey(0)