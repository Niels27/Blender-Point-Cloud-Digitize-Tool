from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS  


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app)

# Define WebSocket events
@socketio.on('message')
def handle_message(data):
    print('received message: ' + str(data))

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/shapefiles/<filename>')
def serve_shapefile(filename):
    return send_from_directory('shapefiles', filename)

if __name__ == '__main__':
    socketio.run(app, debug=True)
