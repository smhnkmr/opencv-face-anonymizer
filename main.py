import cv2

import mediapipe as mp

def process_img(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (50, 50))

    return img

# read image

path = './face.jpg'

img = cv2.imread(path)

H, W, _ = img.shape

type = 'CAM'

# detect faces

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection() as face_detection:
    if type is 'IMG':
        img = process_img(img, face_detection)
        cv2.imshow('img', img)

    elif type is 'VIDEO':
        cap = cv2.VideoCapture('path')
        ret, frame = cap.read()

        while ret:
            frame = process_img(frame, face_detection)
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            ret, frame = cap.read()

        cap.release()

    elif type is 'CAM':
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        while ret:
            frame = process_img(frame, face_detection)
            cv2.imshow('cam', frame)
            cv2.waitKey(25)
            ret, frame = cap.read()

        cap.release()

cv2.waitKey(0)

