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
file_to_read = open("db_geekyfile_friends", "rb")
database = pickle.load(file_to_read)



class VideoProcessor:
    # taking variable input from outside the callback
    def __init__(self) -> None:
        self.iter = np.zeros((1,128))
    def recv(self, frame):
        # step 1 : converting frame ('VideoFrame' Object  of 'pyAV' package) to numpy array
        frm = frame.to_ndarray(format='bgr24')
        y = int(frm.shape[0]/2)
        x = int(frm.shape[1]/2)
        b= int(x/2.5)
        str = ''
        min_dist = 100
        for k in range(25,31,1):
            k = k/10
            a = int(x/k)
            roi = frm[y-a:y+a,x-a:x+a]
            img = cv2.resize(roi, (96,96), interpolation=cv2.INTER_LINEAR)
            img = img[...,::-1]
            img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
            x_train = np.array([img])
            encoding_orig =model.predict_on_batch(x_train)
            self.iter = 0.2*self.iter + (1-0.2)*encoding_orig
            encoding = tf.convert_to_tensor(self.iter, dtype=tf.float32)

            # min_dist = 100
            stri=''
            # Loop over the database dictionary's names and encodings.
            for (name, db_enc) in database.items():
                
                # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
                dist = np.linalg.norm(encoding-db_enc)

                # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
                if dist < min_dist:
                    min_dist = dist
                    identity = name
                    closest_enc = encoding

                # y_origin = (k-2)*100
                if min_dist > 0.5:
                    stri = 'NIB'
                else:
                    dist = dist
                    stri = identity
            # str = str+ '  ' + stri
            str = stri
        ### END CODE HERE ###
        text = 'Please move your face into the black rectangle as shown in above figures.'
        cv2.rectangle(frm, (0,0),(x*2,70),(255,255,255),-1)
        cv2.putText(frm, text, (20,20), font, .5, (0, 0, 0),2)
        cv2.putText(frm, str, (x,50), font, .5, (0, 0, 0),2)
        cv2.rectangle(frm, (x-b, y+b), (x+b, y-b) ,(0,0,0), 2)
        return av.VideoFrame.from_ndarray(frm, format='bgr24')




st.subheader('The App')

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
