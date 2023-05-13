# import streamlit as st

# st.set_page_config(
#     page_title="Trang chủ",
#     page_icon="🇻🇳"
# )

# st.write("# Welcome to Streamlit! 👋")

# st.sidebar.success("Hãy chọn một tính năng.")

# st.markdown(
#     """
#     Streamlit is an open-source app framework built specifically for
#     Machine Learning and Data Science projects.
#     **👈 Select a demo from the sidebar** to see some examples
#     of what Streamlit can do!
#     ### Want to learn more?
#     - Check out [streamlit.io](https://streamlit.io)
#     - Jump into our [documentation](https://docs.streamlit.io)
#     - Ask a question in our [community
#         forums](https://discuss.streamlit.io)
#     ### See more complex demos
#     - Use a neural net to [analyze the Udacity Self-driving Car Image
#         Dataset](https://github.com/streamlit/demo-self-driving)
#     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
# """
# )

import streamlit as st

import cv2

def check_webcam():
    webcam_dict = dict()
    for i in range(0, 10):
        cap = cv2.VideoCapture(i)
        is_camera = cap.isOpened()
        if is_camera:
            webcam_dict[f"index[{i}]"] = "VALID"
            cap.release()
        else:
            webcam_dict[f"index[{i}]"] = None
    return webcam_dict

if __name__ == "__main__":
    st.title('WebCam index validation check')
    webcam_dict = check_webcam()
    st.write(webcam_dict)