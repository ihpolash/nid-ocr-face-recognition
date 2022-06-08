import face_recognition
import cv2
import numpy as np
import os, sys
import pickle

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from detect_id_ocr import *


def face_enrollment(image, name):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    try:
        # change this as you see fit
        try:
            with open('face_encoding.pickle', 'rb+') as handle:
                load_data = pickle.load(handle)

            known_face_encodings = load_data["face_encodings"]
            known_face_names = load_data["user_names"]
        except:
            known_face_encodings = []
            known_face_names = []

        image_path = default_storage.save("tmp/temp.jpg", ContentFile(image.read()))
        tmp_file = os.path.join(settings.MEDIA_ROOT, image_path)

        load_image = face_recognition.load_image_file(tmp_file)
        face_encoding = face_recognition.face_encodings(load_image)[0]

        # Create arrays of known face encodings and their names
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

        store_data = {"face_encodings" : known_face_encodings,"user_names": known_face_names}

        with open(f'face_encoding.pickle', 'wb+') as handle:
            pickle.dump(store_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        default_storage.delete(tmp_file)
        response = {"result": "Enrollment Successful"}
    except:
        response = {"result": "Enrollment Unsuccessful"}

    return response


def face_detect(image):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    try:

        with open('face_encoding.pickle', 'rb+') as handle:
            load_data = pickle.load(handle)

        name = "Unknown"
        known_face_encodings = load_data["face_encodings"]
        known_face_names = load_data["user_names"]
        print(len(known_face_encodings),len(known_face_names))
        print(known_face_names)

        image_path = default_storage.save("tmp/tmp.jpg", ContentFile(image.read()))
        tmp_file = os.path.join(settings.MEDIA_ROOT, image_path)

        img = cv2.imread(tmp_file)

        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        
        default_storage.delete(tmp_file)

        response = {"result": name}
    except:
        response = {"result": "Server Error"}

    return response

def card_ocr(image):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # try:

    image_path = default_storage.save("tmp/temp.jpg", ContentFile(image.read()))
    tmp_file = os.path.join(settings.MEDIA_ROOT, image_path)

    response = run(
        weights=ROOT / 'best_nid.pt',  # model.pt path(s)
        source=ROOT / tmp_file,  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(416, 416),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
    )
    # except:
    #     response = {"result": "OCR Unsuccessful!"}

    return response