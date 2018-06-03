import cv2
import dlib
import pickle
import os
import align
import logging
import threading
import openface
import time
from imutils import face_utils

logger = logging.getLogger(__name__)


detector = dlib.get_frontal_face_detector()

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir,  'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

predictor = "shape_predictor_68_face_landmarks.dat"

networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
imgDim = 96
cuda = False

class FaceRecogniser(object):
    def __init__(self):
        self.align = openface.AlignDlib(os.path.join(dlibModelDir, predictor))
        self.predictor = dlib.shape_predictor(os.path.join(dlibModelDir, predictor))
        self.neuralNetLock = threading.Lock()
        self.net = openface.TorchNeuralNet(networkModel,
                imgDim=imgDim,cuda=cuda)
        logger.info("Opening classifer.pkl")
        self.labels, self.classifier = load_classifier()

    def make_prediction(self,rgbFrame,bb):
        """The function uses the location of a face
        to detect facial landmarks and perform an affine transform
        to align the eyes and nose to the correct positiion.
        The aligned face is passed through the neural net which
        generates 128 measurements which uniquly identify that face.
        These measurements are known as an embedding, and are used
        by the classifier to predict the identity of the person"""

        landmarks = self.align.findLandmarks(rgbFrame, bb)
        if landmarks == None:
            logger.info("///  FACE LANDMARKS COULD NOT BE FOUND  ///")
            return None
        alignedFace = self.align.align(imgDim, rgbFrame, bb,landmarks=landmarks,landmarkIndices=align.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            logger.info("///  FACE COULD NOT BE ALIGNED  ///")
            return None

        logger.info("////  FACE ALIGNED  // ")
        with self.neuralNetLock :
            persondict = self.recognize_face(alignedFace)

        if persondict is None:
            logger.info("/////  FACE COULD NOT BE RECOGNIZED  //")
            return persondict, alignedFace
        else:
            logger.info("/////  FACE RECOGNIZED  /// ")
            return persondict, alignedFace

    def recognize_face(self,img):
        if self.getRep(img) is None:
            return None
        rep1 = self.getRep(img) # Gets embedding representation of image
        logger.info("Embedding returned. Reshaping the image and flatting it out in a 1 dimension array.")
        rep = rep1.reshape(1, -1)   #take the image and  reshape the image array to a single line instead of 2 dimensionals
        start = time.time()
        logger.info("Submitting array for prediction.")
        predictions = self.classifier.predict_proba(rep).ravel() # Computes probabilities of possible outcomes for samples in classifier(clf).
        #logger.info("We need to dig here to know why the probability are not right.")
        maxI = np.argmax(predictions)
        person1 = self.le.inverse_transform(maxI)
        confidence1 = int(math.ceil(predictions[maxI]*100))

        logger.info("Recognition took {} seconds.".format(time.time() - start))
        logger.info("Recognized {} with {:.2f} confidence.".format(person1, confidence1))

        persondict = {'name': person1, 'confidence': confidence1, 'rep':rep1}
        return persondict

    def getRep(self,alignedFace):
        bgrImg = alignedFace
        if bgrImg is None:
            logger.error("unable to load image")
            return None

        logger.info("Tweaking the face color ")
        alignedFace = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        start = time.time()
        logger.info("Getting embedding for the face")
        # rep = self.net.forward(alignedFace) # Gets embedding - 128 measurements
        import face_recognition
        rep = face_recognition.face_encodings(alignedFace) # Gets embedding - 128 measurements
        if len(rep) > 0:
            return rep[0]
        else:
            return None

def load_classifier():
    with open("generated-embeddings/classifier.pkl", 'rb') as f:
        (labels, classifier) = pickle.load(f, encoding='bytes') # Loads labels and classifier SVM or GMM
    return labels, classifier

def find_faces(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb)

    for face in faces:
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 255, 0), 2)

    return

def detect_faces(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb ,1 )
    return faces

def main():
    r = FaceRecogniser()
    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        ret, img_bgr = cap.read()

        find_faces(img_bgr)
#
#        frame = cv2.flip(img_bgr, 1)
#        bb = detect_faces(img_bgr)
#        for face_bb in bb:
#            pred, alignedFace = r.make_prediction(frame, face_bb)
#            print(pred)
#
        # Display the img_bgr
        cv2.imshow('img_bgr',img_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

main()
