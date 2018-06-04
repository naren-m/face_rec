import dlib
import time
import os
import cv2
import logging
import csv
import numpy as np

import align
import model

logging.basicConfig(filename="facenet_training.log", level = logging.INFO)

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
def show_image(window_name, image):
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

# Generate face embeddings/encodings
# TODO: Crate a similar api for batch predictions
def img_to_encoding(aligned_image):
    model = load_model()
    encoding = model.predict(np.array([aligned_image]))

    return encoding[0]

# Get bounding boxes
def detect_faces(image):
    bbs = detector(image, 1)
    return bbs


class Metadata(object):
    def __init__(self, base, name, file_name):
        self.base = base
        self.name = name
        self.file = file_name
        self.image = None
        self.aligned_face = None
        self.encoding = None

    def __iter__(self):
        return iter([self.name, self.encoding])

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)

# Get all image filepaths.
def load_image_paths(path):
    data = list()
    for folder in os.listdir(path):
        for f in os.listdir(os.path.join(path, folder)):
                data.append(Metadata(path, folder, f))
    return data


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
    image_path = 'training-images/Naren/face_12036.jpg'
    rgb_image = read_image_for_dlib(image_path)

    show_image("aligned image", rgb_image)
    bounding_box = detect_faces(rgb_image)
    aligned_image = align_image(rgb_image)
    encoding = img_to_encoding(aligned_image)
    print(encoding)

# Load image filepaths to memory and calculate encodings
def generate_encodings_for_images(base_folder, file_name="encodings.csv"):
    logging.info("Loading files from {}".format(base_folder) )
    start = time.time()

    data = load_image_paths(base_folder)

    logging.info("Took {} seconds to load all image paths".format(time.time() - start))

    with open(file_name, 'w') as f:
        wr = csv.writer(f, delimiter = ',')

        for d in data:
            d.image = read_image_for_dlib(d.image_path())
            d.aligned_face = align_image(d.image)
            start = time.time()
            try:
                d.encoding = img_to_encoding(d.aligned_face)
            except:
                logging.warning("something wrong with image {}".format(d.image_path()))
            logging.info("Took {} seconds generate embeding for {}".format(time.time() - start, d.image_path()))

            wr.writerow([d.image_path(), d.name, d.encoding])

    return data

def save_embedings_to_file():
    data = generate_encodings_for_images("training-images")

save_embedings_to_file()
