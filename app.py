import streamlit as st
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import io
import os
import shutil
import utils as utils
from turn import get_ice_servers


MODEL_NAME = "sign_detection.pt"
IMAGE_ADDRESS = "https://d1v9pyzt136u2g.cloudfront.net/blog/wp-content/uploads/2021/10/31132411/ASL_CHSP_Apr2020.jpeg"
RADIO_IMAGE = "Image"
RADIO_VIDEO = "Video"
RADIO_WEBCAM = "Webcam"
USER_IMAGE_NAME = "user_input.png"
USER_VIDEO_NAME = "user_video.avi"
PREDICTION_PATH = "runs/detect/predict"
PREDICTION_NEW_PATH = "predictions"
PREDICTION_IMAGE_PATH = PREDICTION_NEW_PATH + "/" + USER_IMAGE_NAME
PREDICTION_VIDEO_PATH = PREDICTION_NEW_PATH + "/" + USER_VIDEO_NAME
FINAL_PREDICTION_VIDEO = "output.mp4"
VIDEO_EXTENSION = ".avi"



#load the pytorch weights
model = utils.load_model(MODEL_NAME)

#get predictions
def get_predictions(source_path):
    model.predict(source_path, save = True)
    print("Prediction Complete")

    #copy the predictions and save it independently
    #this is mainly to avoid prediction tree structure
    if os.path.exists(PREDICTION_PATH):
        for item in os.listdir(PREDICTION_PATH):
            shutil.copy(os.path.join(PREDICTION_PATH, item), os.path.join(PREDICTION_NEW_PATH, item))
            if item.endswith(VIDEO_EXTENSION):
                utils.convert_mp4_H264(os.path.join(PREDICTION_NEW_PATH, item), os.path.join(PREDICTION_NEW_PATH, FINAL_PREDICTION_VIDEO))
            os.remove(os.path.join(PREDICTION_PATH, item))

    #then remove the predict director
    os.rmdir(PREDICTION_PATH)
    print("Folder removed!")

    return True


#web application
st.title("American Sign Language Detection")

#set an image
st.image(IMAGE_ADDRESS , caption = "Sign Language Detection")

#detection dashboard
st.header("Detection Dashboard 📷")


#sidebar
with st.sidebar:
    results = False
    video_results = False
    user_image = None
    user_video = None
    #set a header
    st.header("Sign Language Detection")

    #choose a method
    st.subheader("Select a method")

    #set a radio button
    option = st.radio("Select an input option", [RADIO_IMAGE , RADIO_VIDEO, RADIO_WEBCAM], captions=["Detect signs on Images", "Detect signs on videos", "Real time detection"])

    #if image
    if option == RADIO_IMAGE:
        user_image = st.file_uploader("Upload an image", accept_multiple_files=False, help = "Upload any image from your local", type = ["png", "jpg", "jpeg"])

        if user_image:
            if st.button("Detect Signs", use_container_width = True, type = 'primary'):
                results = get_predictions(USER_IMAGE_NAME)

    if option == RADIO_VIDEO:
        user_video = st.file_uploader("Upload a video", accept_multiple_files=False, help = "Upload any video from your local", type = ["mp4", "avi", "mpeg"])
        if user_video:
            if st.button("Detect Signs", use_container_width = True, type = 'primary'):
                with st.spinner("Processing the Video...."):
                    video_results = get_predictions(USER_VIDEO_NAME)


#create two columns
col1, col2 = st.columns(2)

if user_image:

    with col1:
        st.subheader("User Input")
        #set the user image
        st.image(user_image)

        #read and save the image
        image_bytes = io.BytesIO(user_image.read())
        input_image = Image.open(image_bytes)
        input_image.save(USER_IMAGE_NAME)

if results:
    with col2:
        st.subheader("Prediction")
        #set the user image
        st.image(PREDICTION_IMAGE_PATH)

if user_video:
    #set the user video
    st.subheader("User Input Video")
    st.video(user_video)

    #read bytes
    video_bytes = io.BytesIO(user_video.read())

    with open(USER_VIDEO_NAME, "wb") as video_file:
        video_file.write(video_bytes.read())

    video_file.close()

if video_results:
    st.subheader("Predicted Video")
    st.video(os.path.join(PREDICTION_NEW_PATH, FINAL_PREDICTION_VIDEO))

if option == RADIO_WEBCAM:
    conf = 0.2
    webrtc_streamer(
        key="example",
        mode = WebRtcMode.SENDRECV,
        video_processor_factory=lambda : utils.MyVideoTransformer(conf,model),
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": True, "audio": False},
        async_processing  =True
    )
