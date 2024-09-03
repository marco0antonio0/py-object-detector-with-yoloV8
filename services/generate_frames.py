from .get_video_capture import get_video_capture
import time
from .process_frame import process_frame
import cv2
from .draw_layout import draw_layout

def generate_frames(video_source, model, custom_labels, update_interval=0.1):
    video_capture = get_video_capture(video_source)
    last_update_time = time.time()

    previous_boxes = []
    previous_labels = []
    previous_confs = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            previous_boxes, previous_labels, previous_confs = process_frame(frame, model, custom_labels)
            last_update_time = current_time

        draw_layout(frame, previous_boxes, previous_labels)

        # Codificar o frame em formato JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concatena os bytes do frame JPEG

    video_capture.release()