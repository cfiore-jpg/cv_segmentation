import os
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
#from cv_segmentation.layerReplacement.layer_replacement import *


UPLOAD_FOLDER = os.getcwd()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'} #additional extensions, .mp4, .mov, .tiff, .avi



# Takes in a filename and verifies whether it is in the bounds of the allowed
# extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def download_file(name):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], name)

# Uploads the primary or secondary inputs
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        have_file = False
        have_secondary = False
        if 'file' in request.files:
            have_file = True
            file = request.files['file']
        if 'secondary_input' in request.files:
            have_secondary = True
            secondary_input = request.files['secondary_input']
        if not have_file and not have_secondary:
            flash('No file part')
            return redirect(request.url)
        
        #handles primary input uploading
        if have_file:
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                new_path = os.path.join(app.config['UPLOAD_FOLDER'], "static/data/primary_inputs", filename)
                file.save(new_path)
                #url_path = os.path.join("data/uploads", filename)
                #return redirect(url_for('download_file', name=url_path))
                return "File uploaded to static/data/primary_inputs"
        #handles secondary input uploading
        if have_secondary:
            if secondary_input.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if secondary_input and allowed_file(secondary_input.filename):
                filename = secure_filename(secondary_input.filename)
                new_path = os.path.join(app.config['UPLOAD_FOLDER'], "static/data/secondary_inputs", filename)
                secondary_input.save(new_path)
                #url_path = os.path.join("data/uploads", filename)
                #return redirect(url_for('download_file', name=url_path))
                return "File uploaded to static/data/secondary_inputs"
    return

def display_list(list):
    return list

def replace_layers(list):
    return list

@app.route('/', methods=['GET', 'POST'])
# Runs the upload method and returns the rendered page
def index():
    upload_return = upload_file()
    list = display_list(["car", "dog", "tree", "house", "sidewalk", "bike", "crosswalk", "pedestrian", "garage", "bush", "lawn"])
    return render_template('index.html', upload_return=upload_return, list_len=len(list), list=list)

#2.) Displaying list of layers with options for secondary input

#need to figure out how to link images in the secondary_inputs folder with the layer
#they need to replace

#add segment button, write todo w/ empty function




