from flask import Flask, send_from_directory, jsonify, render_template
import os
from glob import glob
import time

app = Flask(__name__)

#Serve  main HTML page
@app.route('/')
def index():
    return render_template('index.html')

#Endpoint to get the latest OBJ and MTL files
@app.route('/get-latest-shape')
def get_latest_shape():
    obj_files = glob('static/objects/*.obj')
    mtl_files = glob('static/objects/*.mtl')
    
    #Sorting files based on modification time
    obj_files.sort(key=os.path.getmtime, reverse=True)
    mtl_files.sort(key=os.path.getmtime, reverse=True)

    #Assuming the newest .obj and .mtl files 
    latest_obj = os.path.basename(obj_files[0]) if obj_files else ''
    latest_mtl = os.path.basename(mtl_files[0]) if mtl_files else ''

    return jsonify({'obj': latest_obj, 'mtl': latest_mtl})


if __name__ == '__main__':
    app.run(debug=True)
