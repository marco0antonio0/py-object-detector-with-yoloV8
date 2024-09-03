from flask import Flask, Response, render_template
from services import draw_layout, get_video_capture, load_model, process_frame, cv2, time
from services.generate_frames import generate_frames

app = Flask(__name__)

@app.route('/camera/0')
def video_feed_local():
    model_path = 'runs/custom_project/my_experiment17/weights/best.pt'
    model = load_model(model_path)
    custom_labels = ['bolinha', 'brinquedo caro', 'cachorro', 'cenora de brinquedo', 'garrafinha de agua', 'humano', 'racao']
    
    return Response(generate_frames(0, model, custom_labels), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/1')
def video_feed_ip():
    model_path = 'runs/custom_project/my_experiment17/weights/best.pt'
    model = load_model(model_path)
    custom_labels = ['bolinha', 'brinquedo caro', 'cachorro', 'cenora de brinquedo', 'garrafinha de agua', 'humano', 'racao']
    
    # Substitua pela URL da sua câmera IP
    ip_camera_url = 'http://192.168.1.147:8080/video'
    return Response(generate_frames(ip_camera_url, model, custom_labels), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
