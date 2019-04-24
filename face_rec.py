class Face:
    def __init__(self, baseImage, face_location, identity="UNKNOWN"):
        self.base = baseImage
        self.location = face_location
        self.identity = identity
        self.image = self._getFace()

    def _getFace(self):
        top, right, bottom, left = self.location
        crop_img = self.base.image[top:bottom, left:right]
        return crop_img

    def save(self, path=''):
        _imageBaseName, _imageExt = os.path.splitext(self.base.imagePath)
        fileName = _imageBaseName.split('/')[-1] + "_face_" + str(
            time.time()) + _imageExt.lower()
        name = os.path.join(path, fileName)
        cv2.imwrite(name, self.image)


class Image:
    def __init__(self, path):
        self.imagePath = path
        self.image = cv2.imread(self.imagePath)
        cv2.flip(self.image, 1)
        self.faces = list()
        self._faceLocations = None

    def resize(self, pixelWidth=1000.0):
        # Resizing image frp pyimage search
        # https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/

        # we need to keep in mind aspect ratio so the image does
        # not look skewed or distorted -- therefore, we calculate
        # the ratio of the new image to the old image
        r = pixelWidth / self.image.shape[1]
        dim = (int(pixelWidth), int(self.image.shape[0] * r))

        # perform the actual resizing of the image and show it
        cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)

    def valid(self):
        if self.image is None or len(self.image.shape) < 3 or len(
                self.image) == 0:
            return False

        return True

    def _getFaceLocations(self):
        return face_recognition.face_locations(self.image)

    def detectFaces(self, resize=False):
        if resize:
            self.resize()
        if not self.valid():
            return
        self._faceLocations = self._getFaceLocations()

        for face_location in self._faceLocations:
            self.faces.append(Face(self, face_location))

    def drawBoundingBoxes(self):
        for face_location in self._faceLocations:
            top, right, bottom, left = face_location
            cv2.rectangle(self.image, (left, top), (right, bottom),
                          (0, 0, 255), 9)

    def saveFaces(self, path='data/faces'):
        for face in self.faces:
            face.save(path)
