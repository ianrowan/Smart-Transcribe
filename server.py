from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import json

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

def transcript():
    while True:
        f = open("transcript.txt", "r")
        content = f.read()
        socketio.emit('transcript', content)
        f.close()
        time.sleep(5)

def definitions():
    while True:
        with open('definitions.txt', 'r') as f:
            data = {}
            for line in f:
                if line.strip():
                    key, value = line.strip().split(':')
                    data[key.strip()] = value.strip()
            json_data = json.dumps(data)
        socketio.emit('definition', json_data)
        time.sleep(5)

if __name__ == '__main__':
    socketio.start_background_task(transcript)
    socketio.start_background_task(definitions)
    socketio.run(app, host="127.0.0.1", port="5050")