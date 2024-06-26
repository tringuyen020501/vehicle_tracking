

# import cv2
# import time
# import torch
# import numpy as np
# import pandas as pd
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from ultralytics import YOLO
# from vidgear.gears import CamGear
# import streamlit as st
# from PIL import Image
# import folium
# from streamlit_folium import st_folium
# import matplotlib.pyplot as plt
# import os

# # Sidebar của Streamlit để nhập URL
# st.sidebar.title("Enter URL Stream YouTube")
# youtube_url = st.sidebar.text_input("Enter the YouTube URL", "https://www.youtube.com/watch?v=fW5e8xsLnBcc&ab_channel=Ph%C3%A1tTri%E1%BB%83n%C4%90%C3%A0N%E1%BA%B5ng")

# # Điều khiển tốc độ phát lại
# playback_speed = st.sidebar.slider("Playback speed ", 0.5, 2.0, 1.0)

# # Các nút điều khiển video
# if st.sidebar.button("Start"):
#     start_stream = True
# else:
#     start_stream = False

# if st.sidebar.button("Stop"):
#     start_stream = False

# if st.sidebar.button("Reset"):
#     start_stream = True

# # Chức năng chụp ảnh
# # if st.sidebar.button("Chụp ảnh"):
# #     capture_snapshot = True
# # else:
# #     capture_snapshot = False

# # Giá trị cấu hình
# conf_threshold = 0.5
# tracking_class = None

# # Khởi tạo DeepSort
# tracker = DeepSort(max_age=10, max_cosine_distance=0.7, max_iou_distance=0.4, n_init=10)

# # Khởi tạo YOLOv8
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = "mps"
# else:
#     device = torch.device("cpu")

# def load_model():
#     model = YOLO("training/models/yolov8n_exp_2.pt") 
#     if torch.cuda.is_available():
#         model = model.to(device)
#     return model

# model = load_model()

# # Tải tên lớp từ tệp
# with open("app/data_ext/classes.names") as f:
#     class_names = f.read().strip().split('\n')

# colors = np.random.randint(0, 255, size=(len(class_names), 3))
# tracks = []

# options = {"STREAM_RESOLUTION": "480p", "CAP_PROP_FPS": 40, "CAP_PROP_BUFFERSIZE": 120}

# # DataFrame để lưu trữ số lượng đối tượng cho biểu đồ đường
# df_counts = pd.DataFrame(columns=['time', 'oto', 'xe2banh'])

# st.markdown(
#     """
#     <style>
#     .info-text {
#         font-size: 18px;
#         margin-top: 60px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Hàm hiển thị bản đồ Google với một điểm đánh dấu
# def display_map():
#     m = folium.Map(location=[16.07421,108.21647], zoom_start=12)
#     folium.Marker(
#         location=[16.07421,108.21647],
#         popup="Click to start streaming",
#         icon=folium.Icon(color="red", icon="info-sign")
#     ).add_to(m)
#     return m
# output_folder = "captured_images"
# # Sử dụng CamGear để stream video từ YouTube nếu nút bắt đầu được nhấn
# if start_stream:
#     # Cài đặt hiển thị của Streamlit
#     st.title("Count and track traffic in a YouTube stream")
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     def stream_video():
#         global df_counts, capture_snapshot  # Truy cập biến df_counts toàn cục

#         capture_snapshot = False
#         snapshot_taken = False
#         frame_placeholder = st.empty()
        

#         # Vòng lặp vô hạn để xử lý từng khung hình
#         while True:
            
#             # Sử dụng CamGear để stream video từ YouTube
#             stream = CamGear(source=youtube_url, stream_mode=True, logging=False, **options).start()

#             col1, col2 = st.columns([4, 1])
#             frame_placeholder = col1.empty()
#             chart_placeholder = col1.empty()
#             info_placeholder = col2.empty()

#             frame_number = 0
#             start_time = pd.Timestamp.now().timestamp()
#             count_vehicle = {class_names.index('oto'): [], class_names.index('xe2banh'): []}
#             while True:
#                 # Đọc nhiều khung hình
#                 frames = [stream.read() for i in range(12)]  # Đọc 12 khung hình cùng lúc
                
#                 # Kiểm tra nếu có khung hình nào là None
#                 if any(frame is None for frame in frames):
#                     # Nếu có, thoát khỏi vòng lặp vô hạn
#                     break

#                 # Điều chỉnh tốc độ đọc khung hình
#                 time.sleep(1 / playback_speed)

#                 # Phát hiện đối tượng bằng mô hình
#                 results = model(frames, stream=True, device=device)

#                 for j, result in enumerate(results):
#                     frame_number += 1
#                     detect = []
#                     for detect_object in result.boxes:
#                         label, confidence, bbox = detect_object.cls, detect_object.conf, detect_object.xyxy[0]
#                         x1, y1, x2, y2 = map(int, bbox)
#                         class_id = int(label)

#                         if tracking_class is None:
#                             if confidence < conf_threshold:
#                                 continue
#                         else:
#                             if class_id != tracking_class or confidence < conf_threshold:
#                                 continue

#                         detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

#                     # Cập nhật các đối tượng theo dõi bằng DeepSort
#                     tracks = tracker.update_tracks(detect, frame=frames[j])  # Có thể chọn bất kỳ khung hình nào để theo dõi

#                     # Vẽ khung hình và ID lên khung hình
#                     class_counts = {class_names.index('oto'): 0, class_names.index('xe2banh'): 0}
#                     for track in tracks:
#                         if track.is_confirmed():
#                             track_id = track.track_id

#                             # Lấy tọa độ và class_id để vẽ lên hình ảnh
#                             ltrb = track.to_ltrb()
#                             class_id = track.get_det_class()
#                             x1, y1, x2, y2 = map(int, ltrb)
#                             color = colors[class_id]
#                             B, G, R = map(int, color)

#                             label = "{}-{}".format(class_names[class_id], track_id)

#                             if class_id == class_names.index('oto'):
#                                 count_vehicle[class_names.index('oto')].append(track_id)
#                                 count_vehicle[class_names.index('oto')] = list(set(count_vehicle[class_names.index('oto')]))
#                             elif class_id == class_names.index('xe2banh'):
#                                 count_vehicle[class_names.index('xe2banh')].append(track_id)
#                                 count_vehicle[class_names.index('xe2banh')] = list(set(count_vehicle[class_names.index('xe2banh')]))

#                             # Vẽ khung hình và nhãn
#                             cv2.rectangle(frames[j], (x1, y1), (x2, y2), (B, G, R), 2)
#                             cv2.rectangle(frames[j], (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
#                             cv2.putText(frames[j], label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#                             # Đếm số lượng đối tượng theo lớp
#                             if class_id in class_counts:
#                                 class_counts[class_id] += 1

#                     # Thêm số liệu vào DataFrame
#                     new_row = {'time': frame_number, 'oto': class_counts[class_names.index('oto')], 'xe2banh': class_counts[class_names.index('xe2banh')]}
#                     df_counts = pd.concat([df_counts, pd.DataFrame([new_row])], ignore_index=True)

#                     # Hiển thị số lượng đối tượng theo lớp
#                     info_text1 = ""
#                     info_text2 = ""
#                     for class_id, count in class_counts.items():
#                         class_name = class_names[class_id]
#                         info_text1 += f"{class_name}: {count}  <br>"
#                         info_text2 += f"Total {class_name}: {len(count_vehicle[class_id])}  <br>"

#                     info_text = info_text1 + "<br>" + info_text2 
#                     info_placeholder.markdown(f"<p class='info-text'>{info_text}</p>", unsafe_allow_html=True) 
                    
                    

#                     # Chụp ảnh nếu nút chụp ảnh được nhấn
#                     img_path = os.path.join(output_folder, f"snapshot_{frame_number}.jpg")
#                     if capture_snapshot and not snapshot_taken:
#                         cv2.imwrite(img_path, frames[j])
#                         print(f"Đã lưu ảnh tại: {img_path}")

#                         capture_snapshot = True
#                         snapshot_taken = True

#                     if not cv2.imwrite(img_path, frames[j]):
#                         print(f"Lỗi khi lưu ảnh tại: {img_path}")


#                     # Hiển thị khung hình
#                     frame_resized = cv2.resize(frames[j], (0, 0), fx=1.2, fy=1.7) 
#                     frame_placeholder.image(frame_resized, channels="BGR", use_column_width=True)    
    

#                 # Cập nhật biểu đồ đường
#                 chart_placeholder.line_chart(df_counts.set_index('time'))

#             # Đóng stream video một cách an toàn
#             stream.stop()

#             # Đóng tất cả các cửa sổ OpenCV
#             cv2.destroyAllWindows()
#     # Gọi hàm để bắt đầu stream video
#     stream_video()

# # Hiển thị bản đồ và kiểm tra nếu điểm đánh dấu được nhấn
# st.title("Count and track traffic in a YouTube stream")
# map_placeholder_folium = st_folium(display_map(), width=700)
# if map_placeholder_folium['last_clicked']:
#     start_stream = True

import streamlit as st
import hashlib
import cv2
import time
import torch
import numpy as np
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from vidgear.gears import CamGear
from PIL import Image
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import os

if 'user_store' not in st.session_state:
    st.session_state['user_store'] = {}

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to check login credentials
def check_login(username, password):
    user_store = st.session_state['user_store']
    hashed_password = hash_password(password)
    if username in user_store and user_store[username] == hashed_password:
        return True
    return False

# Function to handle user registration
def register_user(username, password):
    user_store = st.session_state['user_store']
    if username in user_store:
        return False
    hashed_password = hash_password(password)
    user_store[username] = hashed_password
    return True

# Login and registration page
def login_page():
    st.sidebar.title("Login / Register")
    choice = st.sidebar.selectbox("Choose an option", ["Login", "Register"])

    if choice == "Login":
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if check_login(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.experimental_rerun()
            else:
                st.sidebar.error("Invalid username or password")
    else:
        st.sidebar.subheader("Register")
        username = st.sidebar.text_input("Choose a Username")
        password = st.sidebar.text_input("Choose a Password", type="password")
        if st.sidebar.button("Register"):
            if register_user(username, password):
                st.sidebar.success("User registered successfully! Please login.")
            else:
                st.sidebar.error("Username already exists")

# Streamlit sidebar for YouTube stream URL input
tracks = []
df_counts = pd.DataFrame(columns=['time', 'oto', 'xe2banh'])
def main_page():
    st.sidebar.title("Enter URL Stream YouTube")
    youtube_url = st.sidebar.text_input("Enter the YouTube URL", "https://www.youtube.com/watch?v=fW5e8xsLnBcc&ab_channel=Ph%C3%A1tTri%E1%BB%83n%C4%90%C3%A0N%E1%BA%B5ng")
    playback_speed = st.sidebar.slider("Playback speed ", 0.5, 2.0, 1.0)

    if st.sidebar.button("Start"):
        start_stream = True
    else:
        start_stream = False

    if st.sidebar.button("Stop"):
        start_stream = False

    if st.sidebar.button("Reset"):
        start_stream = True

    conf_threshold = 0.5
    tracking_class = None

    tracker = DeepSort(max_age=10, max_cosine_distance=0.7, max_iou_distance=0.4, n_init=10)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = torch.device("cpu")

    def load_model():
        model = YOLO("training/models/yolov8n_exp_2.pt")
        if torch.cuda.is_available():
            model = model.to(device)
        return model

    model = load_model()

    with open("app/data_ext/classes.names") as f:
        class_names = f.read().strip().split('\n')

    colors = np.random.randint(0, 255, size=(len(class_names), 3))
    
    options = {"STREAM_RESOLUTION": "480p", "CAP_PROP_FPS": 40, "CAP_PROP_BUFFERSIZE": 120}

    st.markdown(
        """
        <style>
        .info-text {
            font-size: 18px;
            margin-top: 60px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    def display_map():
        m = folium.Map(location=[16.07421, 108.21647], zoom_start=12)
        folium.Marker(
            location=[16.07421, 108.21647],
            popup="Click to start streaming",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        return m

    if start_stream:
        st.title("Count and track traffic in a YouTube stream")

        def stream_video():
            global df_counts, capture_snapshot

            capture_snapshot = False
            frame_placeholder = st.empty()

            if 'df_counts' not in st.session_state:
                st.session_state['df_counts'] = pd.DataFrame(columns=["time", "oto", "xe2banh"])

            df_counts = st.session_state['df_counts']
    
            while True:
                stream = CamGear(source=youtube_url, stream_mode=True, logging=False, **options).start()

                col1, col2 = st.columns([4, 1])
                frame_placeholder = col1.empty()
                chart_placeholder = col1.empty()
                info_placeholder = col2.empty()

                frame_number = 0
                # start_time = pd.Timestamp.now().timestamp()
                count_vehicle = {class_names.index('oto'): [], class_names.index('xe2banh'): []}
                while True:
                    frames = [stream.read() for i in range(12)]

                    if any(frame is None for frame in frames):
                        break

                    time.sleep(1 / playback_speed)

                    results = model(frames, stream=True, device=device)

                    for j, result in enumerate(results):
                        frame_number += 1
                        detect = []
                        for detect_object in result.boxes:
                            label, confidence, bbox = detect_object.cls, detect_object.conf, detect_object.xyxy[0]
                            x1, y1, x2, y2 = map(int, bbox)
                            class_id = int(label)

                            if tracking_class is None:
                                if confidence < conf_threshold:
                                    continue
                            else:
                                if class_id != tracking_class or confidence < conf_threshold:
                                    continue

                            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

                        tracks = tracker.update_tracks(detect, frame=frames[j])

                        class_counts = {class_names.index('oto'): 0, class_names.index('xe2banh'): 0}
                        for track in tracks:
                            if track.is_confirmed():
                                track_id = track.track_id
                                ltrb = track.to_ltrb()
                                class_id = track.get_det_class()
                                x1, y1, x2, y2 = map(int, ltrb)
                                color = colors[class_id]
                                B, G, R = map(int, color)

                                label = "{}-{}".format(class_names[class_id], track_id)

                                if class_id == class_names.index('oto'):
                                    count_vehicle[class_names.index('oto')].append(track_id)
                                    count_vehicle[class_names.index('oto')] = list(set(count_vehicle[class_names.index('oto')]))
                                elif class_id == class_names.index('xe2banh'):
                                    count_vehicle[class_names.index('xe2banh')].append(track_id)
                                    count_vehicle[class_names.index('xe2banh')] = list(set(count_vehicle[class_names.index('xe2banh')]))

                                if class_id in class_counts:
                                    class_counts[class_id] += 1


                                cv2.rectangle(frames[j], (x1, y1), (x2, y2), (B, G, R), 2)
                                cv2.rectangle(frames[j], (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
                                cv2.putText(frames[j], label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                                
                        new_row = {'time': frame_number, 'oto': class_counts[class_names.index('oto')], 'xe2banh': class_counts[class_names.index('xe2banh')]}
                        df_counts = pd.concat([df_counts, pd.DataFrame([new_row])], ignore_index=True)

                        info_text1 = ""
                        info_text2 = ""
                        for class_id, count in class_counts.items():
                            class_name = class_names[class_id]
                            info_text1 += f"{class_name}: {count}  <br>"
                            info_text2 += f"Total {class_name}: {len(count_vehicle[class_id])}  <br>"

                        info_text = info_text1 + "<br>" + info_text2 
                        info_placeholder.markdown(f"<p class='info-text'>{info_text}</p>", unsafe_allow_html=True)

                        

                        frame_resized = cv2.resize(frames[j], (0, 0), fx=1.2, fy=1.7) 
                        frame_placeholder.image(frame_resized, channels="BGR", use_column_width=True)

                        chart_placeholder.line_chart(df_counts.set_index('time'))


                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            # start_stream = False
                            break  

                stream.stop()
                cv2.destroyAllWindows()

        stream_video()

    if not start_stream:
        # folium_map = display_map()
        # st_folium(folium_map, width=700)
        st.title("Count and track traffic in a YouTube stream")
        map_placeholder_folium = st_folium(display_map(), width=700)
        if map_placeholder_folium['last_clicked']:
            start_stream = True

if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = None

    if st.session_state['logged_in']:
        main_page()
    else:
        login_page()




