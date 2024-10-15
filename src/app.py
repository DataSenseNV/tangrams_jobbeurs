import streamlit as st
import cv2
from image_processing import Camera


def define_page():
    pass


def click_button():
    st.session_state.is_capturing = not st.session_state.is_capturing


def main(camera: Camera):

    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Webcam Display Streamlit App")
    st.caption("Powered by OpenCV, Streamlit")
    frame_placeholder = st.empty()

    if "last_frame" in st.session_state and st.session_state.last_frame is not None:

        print("processing.....")
        st.title("Last Frame")

        print(st.session_state.last_frame)
        frame_placeholder_last_frame = st.empty()
        frame_placeholder_last_frame.image(st.session_state.last_frame, channels="RGB")
        st.session_state.last_frame = None

        # st.rerun()

    if len(camera.available_cameras) == 0:
        st.error("No cameras detected.")
        return

    # Allow the user to select a camera
    camera.set_active_camera(st.selectbox("Select a camera:", camera.available_cameras))

    button_text = "Stop" if st.session_state.is_capturing else "Start"
    st.button(button_text, on_click=click_button)

    camera.take_image(st.session_state.is_capturing, frame_placeholder)

    print("last_frame", st.session_state.last_frame)

    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    if "is_capturing" not in st.session_state:
        print("inside")
        st.session_state.is_capturing = True

    camera = Camera()
    main(camera)
