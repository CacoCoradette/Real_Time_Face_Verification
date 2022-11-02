# Importing Custom built fucntions
from utils import max_freq_name, triplet_loss
import pickle

# Deep learning libraries (Tensorflow Framework)
from tensorflow.keras.models import load_model
import tensorflow as tf

# Image Processing Libraries
import cv2

# Library for downloading model from google drive
import gdown

# Library for mathematical operations on multidimensional arrays
import numpy as np

# Libraries required for deployment (Done using Streamlit)
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoHTMLAttributes
import av


#initializing variables
font = cv2.FONT_HERSHEY_SIMPLEX

# importing synthetic 128 dimensional encoding vectors of the dataset 
file_to_read = open("geekyfile_cascade_processed", "rb")
database = pickle.load(file_to_read)

#Loading the required haarcascade_frontalface_default.xml” for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Making the Function decorator to memoize(cache) function execution of loading the model
@st.cache
def load_model_cached():
    model = load_model('nn4.small2.lrn.h5',custom_objects = {'triplet_loss' : triplet_loss,'tf':tf})
    return(model)

# loading saved pre-trained model 
model = load_model('nn4.small2.lrn.h5',custom_objects = {'triplet_loss' : triplet_loss,'tf':tf})


# streamlit-webrtc Callback class for all the required frame processing for object detection
# Input -> frame : input frame captured by webrtc-streamer
# Output -> frame : processed input frame for object detection
class VideoProcessor:

    # defining class objects
    def __init__(self) -> None:
        self.queu = ['!!Setting up!!'] # using queue to store 500 predictions at any time
        self.stri = '' # vaiable to store prediction message
        self.j = 0  # counter to control rate at which frames are processed

    def recv(self, frame):
        
        error_message = 'Face Found'

        # step 1 : converting frame ('VideoFrame' Object  of 'pyAV' package) to numpy array
        frm = frame.to_ndarray(format='bgr24')
        # step 2 : flipping screen horizontally 
        frm = cv2.flip(frm,1)
        # step 3 : creating a copy of frame to use cascade classifier
        img = frm
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Predicting using input frame
        if len(faces)!=0:
            i=1   # counter to predict for only one face in frame
            for (x,y,w,h) in faces: # looping through coordinates of faces prdicted
                if i==0:
                    break
                else :
                    min_dist = 100
                    self.j = self.j + 1
                    # create a square bounding box around predicted face
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    # condition to process only 1 frame out of 10
                    if self.j % 10 ==0 :  
                        # accessing the face part of image
                        roi_color = img[y:y+h, x:x+w]  
                        # resizing according to input size of model
                        img = cv2.resize(roi_color, (96,96), interpolation=cv2.INTER_LINEAR)
                        # converting image to bgr format
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                        # changing the order of image from (width,height,channels) to (channels,height,width)
                        img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
                        # converting cv2 object to array
                        x_train = np.array([img])
                        # finding the encoding vector 
                        encoding = model.predict_on_batch(x_train)
                        
                        # looping through the saved encoding vectors to find best match 
                        for (name, db_enc) in database.items():
                            dist = np.linalg.norm(encoding-db_enc)
                            if dist < min_dist:
                                min_dist = dist
                                identity = name[0:name.find('__')]
                            
                            # predicting unable to recognize is min_dist is less then 0.35
                            if min_dist >= 0.35:
                                if min_dist >= 0.95:
                                    self.stri = 'Sorry!! Unable to recognize :('

                            else:
                                # appending new prediction to queue depending on the size of the queue 
                                if len(self.queu)<=500:
                                    self.queu.append(identity)
                                else :
                                    self.queu.pop(0)
                                    self.queu.append(identity)

                                # giving final prediction based on the max frequency of the last 500 predictions
                                max,res = max_freq_name(self.queu)
                                if max==1:
                                    self.stri = self.queu[0]
                                elif max!=1:
                                    self.stri = res

        else :
            # message if cascade classifier is unable to find a face
            error_message = 'Finding Face!!!!'
        
        # Final display messages and their design
        (label_width, label_height), baseline = cv2.getTextSize(self.stri, font, .5, 1)
        (label_width_1, label_height_1), baseline = cv2.getTextSize(error_message, font, 1, 2)
        cv2.rectangle(frm, (0,0),(frm.shape[1],40),(255,255,255),-1)
        if error_message != 'Finding Face!!!!' : 
            cv2.putText(frm, self.stri, (int(frm.shape[1]/2-label_width/2),20), font, 0.5, (0, 0, 0),1)
        cv2.rectangle(frm,(0,frm.shape[0]-40),(frm.shape[1],frm.shape[0]),(255,255,255),-1)
        cv2.putText(frm, error_message, (int(frm.shape[1]/2-label_width_1/2),frm.shape[0]-1), font, 1, (0, 0, 0),2)

        return av.VideoFrame.from_ndarray(frm, format='bgr24')


#---------------------------------------------------------------------------------------------------------------------------------------------------------

# Streamlit Web-Based Real-Time Video Processing App Hosted on Streamlit Cloud

# Packages Used:
# streamlit : Main Package
# streamlit-webrtc : A custom component of Streamlit which deals with real-time video and audio streams
# opencv : Image processing


#settimng up background image for app
st.markdown(
   f'''
   <style>
   .stApp {{
             background: url("https://img.freepik.com/premium-photo/abstract-communication-technology-network-concept_34629-641.jpg?w=1380");
             background-size: cover
         }}
   </style>
   ''',
   unsafe_allow_html=True)


#creating containers for different sections of app
header = st.container()
app = st.container()
video = st.container()
model_intro = st.container()
model_details = st.container()
references = st.container()


with header:
    st.title('Real Time Face Recognition')
    st.markdown("""---""")

    st.write("Face recognition is a 1:K way of identifying or confirming an individual's identity using their face . Facial Recognition systems can be used to identify photos, videos, or in real time. It has applications in mobile security system, Airport Immigration services, Attendence system, and many more ways.")
    st.markdown("""---""")

with app:

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

    st.markdown("""---""")

with video:

    st.subheader('Sample Video')
    video = open('samplevideo.mkv','rb')
    st.video(video)
    
with model_intro:
    st.subheader('About Model')
    st.write('The model used in this project is highlt inspired from "FaceNet Model".FaceNet  directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. These embeddings can be then used as feature vectors for face recognition or face verification.')

    st.markdown('**YOLO Architecture**')
    st.write("Inspired by the FaceNet architecture, our model uses 3 inception layers with input size of (96,96,3) and output embedding vector of shape(1,1,128).Finally it uses triplet loss function for traning. ")
    st.image('images/inception_network.JPG')
    st.write ('Inception Block')
    st.markdown("""---""")

    st.subheader('About my project')
    st.write('In this project I have used FaceNet model for face verification.I have used 3 to 5 images of each of my friends to create my personal database for verification. I have applied data augmentation to enlarge my dataset such that I was able to produce 6 images from a single image.My data had images of nearly 40 persons and average 4 photos of each. Thus after augmenting I was able to create a dataset of 960.')
    st.markdown("""---""")

with model_details:

    st.subheader('Some indepth Model Details ')

    st.markdown('**Inception Layer**')
    st.text('''
    The key idea for devising this architecture is to deploy multiple convolutions with multiple filters and pooling layers simultaneously in parallel within the same layer (inception layer). For example, in the image shown above, employs convolution with 1x1 filters as well as 3x3 and 5x5 filters and a max pooling layer. Further 1x1 convolution layers can be used for dimensional reduction.
    The intention is to let the neural network learn the best weights when training the network and automatically select the more useful features. Additionally, it intends to reduce the no. of dimensions so that the no. of units and layers can be increased at later stages. The side-effect of this is to increase the computational cost for training this layer. To address this, a number of solutions have been suggested in the paper such as to deploy parallel computations for this architecture.
    ''')

    st.markdown('**Triplet Loss Function**')
    st.text('''
    By using a 128-neuron fully connected layer as its last layer, the model ensures that the output is an encoding vector of size 128.By computing the distance between two encodings and thresholding, we can determine if the two pictures represent the same person or not.So, an encoding is a good one if:
    ~ The encodings of two images of the same person are quite similar to each other.
    ~ The encodings of two images of different persons are very different.
    The triplet loss function formalizes this, and tries to "push" the encodings of two images of the same person (Anchor and Positive) closer together, while "pulling" the encodings of two images of different persons (Anchor, Negative) further apart.

    Training will use triplets (A,P,N) of images :

    ~A is an "Anchor" image--a picture of a person.
    ~P is a "Positive" image--a picture of the same person as the Anchor image.
    ~N is a "Negative" image--a picture of a different person than the Anchor image.
    These triplets are picked from the training dataset. 
    We used this loss fucntion.
    ||f(A)-f(P)||^2 + alpha < ||f(A)-f(N)||^2 

    ''')
    st.image('images/triplet_loss_function.JPG')


with references:
    st.subheader('Refernces')
    checkbox = st.checkbox("Show references")
    if(checkbox):
        st.write("""
        1. Official FaceNet Paper (https://arxiv.org/pdf/1503.03832.pdf).
        2. github repository for getting pretrained model  (https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb).
        3. Coursera DeepLearning Specialization (https://www.coursera.org/specializations/deep-learning) for project implementation.
        4. Dr. Andrew NG (My role Model for ML/DL)
        """)
