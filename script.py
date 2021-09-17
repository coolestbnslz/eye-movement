from flask import *
from main import  *
from flask import Flask, render_template, Response
import cv2
from detect import detector
app = Flask(__name__)
camera = cv2.VideoCapture(0)
@app.route('/')
def message():
    return render_template('web.html')
def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    max_X = 20  # max movement of x possible
    max_Y = 20  # max movement of y possible
    min_jump_X = 600  # not used now
    min_jump_Y = 400
    prev_X, prev_Y = locate_cursor()
    model = load_model()
    dtr = detector()
    while True:
        success, frame = camera.read()
        if success:
            frame = cv2.flip(frame, 1)
            eyes = dtr.detect(frame)
            for (x, y, w, h) in eyes:
                img = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_RGB2GRAY)
                img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.circle(frame,(int(x + w/2),int(y+h/2)),int(h*0.7),(255,0,0),2)
                img = img.reshape((1,) + img.shape + (1,)) / 255.
                y_pred = model.predict(img)  # predict the position
                x = int(y_pred[0][0])
                y = int(y_pred[0][1])
                if x - prev_X > max_X:
                    x = prev_X + max_X
                elif x - prev_X < -max_X:
                    x = prev_X - max_X
                if y - prev_Y > max_Y:
                    y = prev_Y + max_Y
                elif y - prev_Y < -max_Y:
                    y = prev_Y - max_Y
                move(x, y)
                prev_X = x
                prev_Y = y
                break

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)