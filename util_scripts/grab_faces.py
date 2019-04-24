import os
import cv2
import face_recognition
import logging
import shutil, time

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


def main():
    logging.debug("Started processing")

    for walk_dir, dirs, files in os.walk(photo_dir):
        for imagePath in files:
            imagePath = os.path.join(walk_dir, imagePath)

            image = Image(imagePath)
            try:
                # load image with face_recognition
                image.detectFaces()
                image.saveFaces()
                # image.drawBoundingBoxes()
            except Exception as e:
                logging.error("Unable process image file: {}".format(str(e)))

            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)
            try:
                shutil.move(imagePath, processed_dir)
            except Exception as e:
                logging.error("Unable to move file:{}".format(str(e)))


if __name__ == "__main__":
    main()
