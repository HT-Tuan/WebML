import cv2
import numpy as np
import pyvirtualcam

import streamlit as st

# Tạo ảo hóa camera với kích thước 640x480 và tốc độ khung hình 30fps
with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
    # Tạo đối tượng VideoCapture để truy cập camera thật
    cap = cv2.VideoCapture(0)

    # Kiểm tra xem VideoCapture có hoạt động không
    if not cap.isOpened():
        st.error("Không thể truy cập camera thật.")
        st.stop()

    # Vòng lặp để đọc các khung hình từ camera thật và đưa chúng vào camera ảo
    while True:
        # Đọc khung hình mới nhất từ camera thật
        ret, frame = cap.read()

        # Kiểm tra xem VideoCapture có đọc được khung hình hay không
        if not ret:
            st.error("Không thể đọc khung hình từ camera thật.")
            st.stop()

        # Xử lý khung hình ở đây (ví dụ: chuyển đổi sang đen trắng)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Đưa khung hình đã xử lý vào camera ảo
        cam.send(gray_frame)

        # Đọc khung hình mới nhất từ camera ảo
        img = cam.read()

        # Hiển thị khung hình đã xử lý và khung hình từ camera ảo bằng Streamlit
        st.image(np.hstack((gray_frame, img)), channels="GRAY", width=640)

        # Kiểm tra xem người dùng đã nhấn nút "Stop" chưa
        if st.button("Stop"):
            break

# Giải phóng tài nguyên
cap.release()
try:
    cv2.destroyAllWindows()
except cv2.error:
    pass
