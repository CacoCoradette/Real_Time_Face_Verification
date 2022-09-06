import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoHTMLAttributes
import av
import pickle
font = cv2.FONT_HERSHEY_SIMPLEX
import streamlit as st
from triplet_loss import triplet_loss
from tensorflow.keras.models import load_model
import tensorflow as tf
# from database import database


model = load_model('nn4.small2.lrn.h5',custom_objects = {'triplet_loss' : triplet_loss,'tf':tf})
print(model.outputs)
file_to_read = open("geekyfile_cascade_processed", "rb")
database = pickle.load(file_to_read)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class VideoProcessor:
    def __init__(self) -> None:
        self.iter = np.zeros((1,128))
        self.stri = 'Not In DataBase'
    def recv(self, frame):
        alpha = 0.15
        frm = frame.to_ndarray(format='bgr24')
        img = frm
        # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces)!=0:
            i=0
            for (x,y,w,h) in faces:
                if i == 1 :
                    break
                else :
                        i=1
                        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                        roi_color = img[y:y+h, x:x+w]
                        img = cv2.resize(roi_color, (96,96), interpolation=cv2.INTER_LINEAR)
                        img = img[...,::-1]
                        img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
                        x_train = np.array([img])
                        encoding_orig =model.predict_on_batch(x_train)
                        encoding = alpha*self.iter + (1-alpha)*encoding_orig
                        min_dist = 100
                        for (name, db_enc) in database.items():
                            dist = np.linalg.norm(encoding-db_enc)
                            if dist < min_dist:
                                min_dist = dist
                                identity = name[0:name.find('__')]
                                # min_enc = encoding
                            if min_dist > 0.4:
                                # stri = 'Not in DB'
                                continue
                            else:
                                dist = dist
                                self.stri = identity

        cv2.rectangle(frm, (0,0),(frm.shape[1],40),(255,255,255),-1)
        cv2.putText(frm, self.stri, (int(frm.shape[1]/2),20), font, .5, (0, 0, 0),2)
        return av.VideoFrame.from_ndarray(frm, format='bgr24')


# class VideoProcessor:
#     # taking variable input from outside the callback
#     # def __init__(self) -> None:
#     #     self.iter = np.zeros((1,128))
#     def recv(self, frame):
#         # step 1 : converting frame ('VideoFrame' Object  of 'pyAV' package) to numpy array
#         frm = frame.to_ndarray(format='bgr24')
#         y = int(frm.shape[0]/2)
#         x = int(frm.shape[1]/2)
#         b= int(x/2.5)
#         str = ''
#         min_dist = 100
#         for k in range(25,31,1):
#             k = k/10
#             a = int(x/k)
#             roi = frm[y-a:y+a,x-a:x+a]
#             img = cv2.resize(roi, (96,96), interpolation=cv2.INTER_LINEAR)
#             img = img[...,::-1]
#             img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
#             x_train = np.array([img])
#             encoding_orig =model.predict_on_batch(x_train)
#             # self.iter = 0.2*self.iter + (1-0.2)*encoding_orig
#             # encoding = tf.convert_to_tensor(self.iter, dtype=tf.float32)
#             encoding = encoding_orig

#             # min_dist = 100
#             stri=''
#             # Loop over the database dictionary's names and encodings.
#             for (name, db_enc) in database.items():
                
#                 # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
#                 dist = np.linalg.norm(encoding-db_enc)

#                 # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
#                 if dist < min_dist:
#                     min_dist = dist
#                     identity = name[0:name.find('__')]
#                     closest_enc = encoding

#                 # y_origin = (k-2)*100
#                 if min_dist > 0.4:
#                     stri = 'Not in DataBase'
#                 else:
#                     dist = dist
#                     stri = identity
#             str = stri
#         ### END CODE HERE ###
#         cv2.rectangle(frm, (0,0),(x*2,40),(255,255,255),-1)
#         cv2.putText(frm, str, (x,20), font, .5, (0, 0, 0),2)
#         cv2.rectangle(frm, (x-b, y+b), (x+b, y-b) ,(0,0,0), 2)
#         return av.VideoFrame.from_ndarray(frm, format='bgr24')




st.subheader('The App')
# image = cv2.imread('sample.JPG')
# st.image(image)

# streamlit-webrtc requires callbacks to process image and audio frames which is one major
# difference between OpenCV GUI and streamlit-webrtc
ctx = webrtc_streamer(key='key',
                    # class object passed to video_processor_factory for video processing
                    video_processor_factory=VideoProcessor,
                    # rtc_configuration parameter to deploy the app to the cloud
                    rtc_configuration=RTCConfiguration(
                        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                    # muting the audio input
                    video_html_attrs=VideoHTMLAttributes(
                        autoPlay=True, controls=True, style={"width": "100%"}, muted=True),
                    )

# if ctx.video_processor:
#     ctx.video_processor.max_boxes = st.slider(
#         'Max boxes to predict', min_value=1, max_value=10, value=5, step=1)
#     ctx.video_processor.score_threshold = st.slider(
#         'Score Threshold ', min_value=0.0, max_value=1.0, value=.5, step=.1)
st.markdown("""---""")

text = 'Please move your face into the black rectangle as shown in below figures.'
st.write(text)
image = cv2.imread('sample.JPG')
st.image(image)
