import logging
import dlib
import os
import cv2

import numpy as np

import align
import model

# from model import create_model


logger = logging.getLogger(__name__)


detector = dlib.get_frontal_face_detector()

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir,  'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
kerasOpenfaceModelDir = os.path.join(modelDir, 'keras_openface')

predictor = "shape_predictor_68_face_landmarks.dat"

# Load pretrained keras openface model that generates face embeddings
def load_model():
    model_full_pth = os.path.join(kerasOpenfaceModelDir, "nn4.small2.v1.h5")
    nn4_small2_pretrained = model.create_model()
    nn4_small2_pretrained.load_weights(model_full_pth)

    return nn4_small2_pretrained

# Show image in Opencv window
def show_image(image, window_name):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Read image in filepath
def read_image(path):
    image = cv2.imread(path, 1)

    return image

# Read image in Dlib format from filepath
def read_image_for_dlib(path):
    image = cv2.imread(path, 1)

    # Converts the image from BGR (Opencv format) to RGB (Dlib fromat)
    return cv2.flip(image,1)


# Align the face. Input format of the face is Dlib(rgb)
# AlignDlib is a wrapper on top of dlib, to help align the face landmarks.
#   CMU wrote the AlignDlib.
# TODO: Check if needed any modification for my usecase.
def align_image(img):
    alignment = align.AlignDlib(os.path.join(dlibModelDir, predictor))

    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                          landmarkIndices=align.AlignDlib.OUTER_EYES_AND_NOSE)

# Get face landmarks ?
# Generate face embeddings/encodings

# Get bounding boxes
def detect_faces(image):
    bbs = detector(image, 1)
    return bbs

# Load image filepaths to memory
# Train the keral model with new faces
# Train the classifier on face embeddings
# Classify face using the classifier
# Infer on new faces

# Class for Camera

# Class for Face
# Class for Detector
# Class for Embedding
# Class for Recognition

# Playground
def playground():
    image_path = 'images_full/Naren/IMG_0101.jpg'

    rgb_image = read_image_for_dlib(image_path)
    bounding_box = detect_faces(rgb_image)
    aligned_image = align_image(rgb_image)
    # show_image(aligned_image, "aligned image")
    model = load_model()
    encoding = model.predict(np.array([aligned_image]))
    print(encoding)

playground()
