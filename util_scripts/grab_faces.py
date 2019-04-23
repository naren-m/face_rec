import os
import cv2
import face_recognition
import logging
import shutil

from random import *

logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    format=
    "[%(asctime)s] {%(pathname)s} %(funcName)s:%(lineno)d %(levelname)s - %(message)s",
    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

face_count = 0
faces_folder = "data/faces/"

photo_dir = "data/Photos"
# photo_dir = "data/tmp/"

processed_dir = "data/processed/photos"

image_extensions = list()

image_extensions.append(".jpg")
image_extensions.append(".png")
image_extensions.append(".JPG")
image_extensions.append(".PNG")


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
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def crop_image(face_location, image):
    top, right, bottom, left = face_location
    crop_img = image[top:bottom, left:right]
    return crop_img


def valid_image(image):
    if image is None or len(image.shape) < 3 or len(image) == 0:
        return False

    return True


logging.debug("Started processing")

for walk_dir, dirs, files in os.walk(photo_dir):
    for image_name in files:
        _, ext = os.path.splitext(image_name)
        if not ext in image_extensions:
            logging.debug("File {} is not an image".format(image_name))
            continue

        image_name = os.path.join(walk_dir, image_name)

        image = None
        face_locations = []
        try:
            # load image with face_recognition
            image = face_recognition.load_image_file(image_name)
            image = resize_image(image)
            face_locations = face_recognition.face_locations(image)
            if not valid_image(image):
                continue
        except Exception as e:
            logging.error("Unable process image file: {}".format(str(e)))

        for face_location in face_locations:
            top, right, bottom, left = face_location
            try:
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255),
                              2)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = crop_image(face_location, image)
            except Exception as e:
                logging.debug("Error post processing image {}".format(str(e)))
                continue

            if not valid_image(image):
                continue

            face_count += 1
            face_file_name = faces_folder + "face_" + str(randint(
                1, 1000000)) + ext.lower()
            cv2.imwrite(face_file_name, image)

            logging.info(
                "Processed image {} and saved to {}, face count {}".format(
                    image_name, face_file_name, face_count))

        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        try:
            shutil.move(image_name, processed_dir)
        except Exception as e:
            logging.error("Unable to move file:{}".format(str(e)))
