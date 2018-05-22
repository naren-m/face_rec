import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os.path

from align import AlignDlib
import bz2
import os

from keras.models import load_model
from keras.utils import CustomObjectScope

import tensorflow as tf

from urllib.request import urlopen

from model import create_model

EMBEDDING_FIME_NAME = "embeddings.npy"

def save_embeddings(filename, embedded):
    np.save(filename, embedded)

def get_embeddings(filename):
    return np.load(filename)

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    
def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)


def get_model():
    nn4_small2_pretrained = create_model()
    nn4_small2_pretrained.load_weights('model/nn4.small2.v1.h5')
    return nn4_small2_pretrained

def download_landmarks(dst_file):
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    decompressor = bz2.BZ2Decompressor()
    
    with urlopen(url) as src, open(dst_file, 'wb') as dst:
        data = src.read(1024)
        while len(data) > 0:
            dst.write(decompressor.decompress(data))
            data = src.read(1024)

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

# Transform image using specified face landmark indices and crop image to 96x96
def align_image(img):
    alignment = AlignDlib('./models/landmarks.dat')
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
			landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def get_largest_bounding_box(img):
    alignment = AlignDlib('./models/landmarks.dat')
    return alignment.getLargestFaceBoundingBox(img)

def create_embeddings(metadata):
    embedded = np.zeros((metadata.shape[0], 128))
    nn4_small2_pretrained = get_model()
    for i, m in enumerate(metadata):
        
        img = load_image(m.image_path())
        if img is None:
            continue
        img = align_image(img)
        
        try:
            # scale RGB values to interval [0,1]
            img = (img / 255.).astype(np.float32)
            embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
        except Exception as e:
            print(m.file)
            continue
    return embedded

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(embedded, idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle('Distance= {:.2}'.format(distance(embedded[idx1], embedded[idx2]).astype(float)))
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()));   

dst_dir = 'models'
dst_file = os.path.join(dst_dir, 'landmarks.dat')

if not os.path.exists(dst_file):
    os.makedirs(dst_dir)
    download_landmarks(dst_file)

metadata = load_metadata('images')