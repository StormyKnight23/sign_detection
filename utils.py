from moviepy.editor import VideoFileClip
from ultralytics import YOLO
import streamlit as st
from streamlit_webrtc import VideoTransformerBase
import numpy as np
from PIL import Image
import av


def convert_mp4_H264(input_file:str, output_file:str) -> None:

    # Load the video clip
    clip = VideoFileClip(input_file)

    # Set the codec to H.264
    codec = "libx264"

    # Save the video with the specified codec
    clip.write_videofile(output_file, codec=codec)

    print("Conversion complete.")


@st.cache_resource
def load_model(model_path):

    return YOLO(model_path)


class MyVideoTransformer(VideoTransformerBase):
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        processed_image = self._display_detected_frames(image)
        st.image(processed_image, caption='Detected Video', channels="BGR", use_column_width=True)
        return av.VideoFrame.from_ndarray(processed_image, format="bgr24")

    def _display_detected_frames(self, image):
        orig_h, orig_w = image.shape[0:2]
        width = 240  # Set the desired width for processing

        # cv2.resize used in a forked thread may cause memory leaks
        input = np.asarray(Image.fromarray(image).resize((width, int(width * orig_h / orig_w))))

        if self.model is not None:
            # Perform object detection using YOLO model
            res = self.model.predict(input, conf=self.conf)

            # Plot the detected objects on the video frame
            res_plotted = res[0].plot()
            return res_plotted

        return input
