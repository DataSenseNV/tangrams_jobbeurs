import streamlit as st
import cv2
from image_processing import Camera

def define_page():
    pass


def main(camera: Camera):

    

    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Webcam Display Streamlit App")
    st.caption("Powered by OpenCV, Streamlit")
    frame_placeholder = st.empty()

    if 'last_frame' in st.session_state and st.session_state.last_frame is not None:

        print('processing.....')
        print(st.session_state.last_frame)
        frame_placeholder.image(st.session_state.last_frame, channels="RGB")
        st.session_state.last_frame = None

        st.rerun()

   
    
    print('current_index: ',  camera.current_camera_index)

    if len(camera.available_cameras) == 0:
        st.error("No cameras detected.")
        return

    # Allow the user to select a camera
    camera.set_active_camera(st.selectbox("Select a camera:",  camera.available_cameras))


    button_text = "Stop" if st.session_state.is_capturing else "Start"

    st.session_state.is_capturing = not st.button(button_text)

    camera.take_image(st.session_state.is_capturing, frame_placeholder)

    print('last_frame', st.session_state.last_frame)

    

    #cap.release()
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    if 'is_capturing' not in st.session_state:
        print('inside')
        st.session_state.is_capturing = True

    camera = Camera()
    main(camera)
