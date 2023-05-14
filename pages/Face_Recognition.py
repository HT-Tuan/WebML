import streamlit as st

st.set_page_config(
    page_title="Nhận dạng khuôn mặt",
    page_icon="😎"
)

import av
import numpy as np
import streamlit_webrtc as webrtc

def receive_frame(frames):
    for frame in frames:
        # Convert the raw bytes to a numpy array
        img = frame.to_ndarray(format="bgr24")

        # Inference
        tm.start()
        faces = detector.detect(img) # faces is a tuple
        tm.stop()

        # Draw results on the input image
        visualize(img, faces, tm.getFPS())

        # Visualize results
        FRAME_WINDOW.image(img, channels='BGR')

webrtc_ctx = webrtc.WebRtcStreamerContext(
    key="webcam",
    video_transformer_factory=None,
    # Use the receive_frame function to process incoming frames
    client_settings=webrtc.ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    ),
    sendback_audio=False,
    async_processing=True,
    source_video_track_kind="video",
    # Set the receive_frame function as the callback for incoming frames
    on_video_frame=receive_frame
)