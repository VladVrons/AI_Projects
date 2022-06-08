import dlib
from scipy.spatial import distance
from skimage import io
from os import listdir
from os.path import isfile, join

sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()


def descript(imgname):
    img = io.imread(imgname)
    img = dlib.resize_image(img, 700, 500)
    win1 = dlib.image_window()
    win1.clear_overlay()
    win1.set_image(img)

    dets = detector(img, 1)
    for k, d in enumerate(dets):
        print(" Detection {}: Left : {} Top: {} Right : {} Bottom : {}".format(k, d.left(), d.top(), d.right(),
                                                                               d.bottom()))
    shape = sp(img, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)
    win1.wait_until_closed()
    return facerec.compute_face_descriptor(img, shape)


def task1():
    face_descriptor1 = descript('me.jpg')
    face_descriptor2 = descript("also_me.jpg")
    a = distance.euclidean(face_descriptor1, face_descriptor2)
    print(a)


def get_list_Of_Files(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


class Photos:
    def __init__(self, name, descr):
        self.name = name
        self.descriptor = descr


def list_Of_descriptors():
    listP = get_list_Of_Files('politicians')
    dict = []
    for name in listP:
        dict.append(Photos(name, descript('politicians\\' + name)))
    return dict


def testImage():
    img = io.imread('test.jpg ')
    win2 = dlib.image_window()
    win2.set_image(img)
    dets = detector(img, 1)
    dict = list_Of_descriptors()

    for el in dict:
        face_descriptor2 = el.descriptor
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            vidstan = distance.euclidean(face_descriptor2, face_descriptor)
            if vidstan < 0.6:
                print('Знайшов', vidstan, '  ', el.name[:len(el.name) - 4])
                print(" Detection {}: Left : {} Top: {} Right : {} Bottom : {}".format(k, d.left(), d.top(), d.right(),
                                                                                       d.bottom()))
                win2.clear_overlay()
                win2.add_overlay(d)
                win2.add_overlay(shape)


task1()
testImage()
