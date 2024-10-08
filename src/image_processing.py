import cv2
import streamlit as st

class Camera:
    def __init__(self) -> None:
        self.available_cameras = self._get_available_cameras()
        self.current_camera_index = -1
        self.current_video_capture = None
        self.last_frame = None
        if 'last_frame' not in st.session_state:
            st.session_state.last_frame = None

    def _get_available_cameras(self) -> list:
        """Returns a list of available camera device indices."""
        index = 0
        available_cameras = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:  # Check if the camera can capture a frame
                cap.release()
                break
            else:
                available_cameras.append(index)
            cap.release()
            index += 1
        return available_cameras
    
    def set_active_camera(self, index: int):
        if index in self.available_cameras:  # Ensure the camera index is valid
            self.current_camera_index = index
            self.current_video_capture = cv2.VideoCapture(index)
        else:
            st.write(f"Camera with index {index} not available.")
    
    def take_image(self, is_capturing: bool, frame_placeholder):
        if not is_capturing:
            return

        if not self.current_video_capture.isOpened() and is_capturing:
            self.current_video_capture = cv2.VideoCapture(self.current_camera_index)
        
        while is_capturing:
            ret, frame = self.current_video_capture.read()
            if not ret:
                st.write("Video Capture Ended")
                break

            # Convert the frame from BGR (OpenCV default) to RGB (for Streamlit display)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")

            st.session_state.last_frame = frame

    def process(image):
        pass

