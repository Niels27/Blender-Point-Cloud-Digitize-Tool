from flask import Flask, send_from_directory, render_template, Response
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get-point-cloud')
def get_point_cloud():
    
    file_path = '/Users/niels/Documents/GitHub/Blender-Point-Cloud-Digitize-Tool/JSON files/auto.laz_points_colors_uncompressed.json'  

    # Check if the file is gzip compressed (based on file extension)
    if file_path.endswith('.gz'):
        # Serve gzip compressed file
        with open(file_path, 'rb') as file:
            gzip_data = file.read()
        response = Response(gzip_data)
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Type'] = 'application/json'
        return response
    else:
        # Serve regular JSON file
        directory = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        return send_from_directory(directory, file_name, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
