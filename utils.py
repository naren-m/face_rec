import cv2
import os
import face_recognition
from PIL import Image


def read_image(file_name):
    image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    return image


def resize_image(image):
    # Resizing image frp pyimage search
    # https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/

    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    pixel_width = 1000.0
    r = pixel_width / image.shape[1]
    dim = (int(pixel_width), int(image.shape[0] * r))
 
    # perform the actual resizing of the image and show it
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_image(face_location, image):
    top, right, bottom, left = face_location
    crop_img = image[top:bottom, left:right] # Crop from x, y, w, h -> 100, 200, 300, 400
    return crop_img

faces_folder = "data/faces/"
face_count = 0
image_name= "data/Photos/Naren_Marraige_Dumps/Rohit_marriage_pics/Naren_Solo_Pics/2015_11_26_07_09_50.JPG"
f, ext = os.path.splitext(image_name)
image = face_recognition.load_image_file(image_name)
# image = resize_image(image)
face_locations = face_recognition.face_locations(image)



for face_location in face_locations:
    # Print the location of each face in this image
    print(face_location)
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image(face_location, image)
    face_count += 1
    face_file_name = faces_folder + "face_" + str(face_count) + ext.lower()
    cv2.imwrite(face_file_name, image)
    show_image(image)




