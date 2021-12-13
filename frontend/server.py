import os
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
#from cv_segmentation.layerReplacement.processing import test_pims

# from cv_segmentation.layerReplacement.layer_replacement import *


UPLOAD_FOLDER = os.getcwd()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}  # additional extensions, .mp4, .mov, .tiff, .avi
primary_input = None


# Takes in a filename and verifies whether it is in the bounds of the allowed
# extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Uploads the primary input
def upload_primary_input():
    global primary_input
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                new_path = os.path.join(app.config['UPLOAD_FOLDER'], "static/data/primary_inputs", filename)
                file.save(new_path)
                primary_input = new_path
                return new_path
    return

# Uploads secondary inputs
def upload_secondary_input():
    if request.method == 'POST':
        if 'secondary_input' in request.files:
            secondary_input = request.files['secondary_input']
            if secondary_input.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if secondary_input and allowed_file(secondary_input.filename):
                filename = secure_filename(secondary_input.filename)
                new_path = os.path.join(app.config['UPLOAD_FOLDER'], "static/data/secondary_inputs", filename)
                secondary_input.save(new_path)
                return new_path
    return

def do_segment():
    global counter

    #TODO: Check if there is an input image, this is not tested yet
     # might have to pull it from the folder and only allow one file at a time

   # if primary_input:
    if request.method == 'POST':
        print("rf", request.form)
        if 'segment_button' in request.form: 
            if request.form['segment_button'] == 'Segment':
                print("pi",primary_input)

                #TODO: test_pims function here. This should in some way return the labels needed for frontend.
                temp_list = ["car", "dog", "tree", "house", "sidewalk", "bike", "crosswalk", "pedestrian", "garage", "bush", "lawn"]
                return temp_list

def display_list(list):
    return list


def replace_layers(list):
    return list


@app.route('/', methods=['GET', 'POST'])
# Runs the upload method and returns the rendered page
def index():
    upload_primary_input()
    upload_secondary_input()
    list = do_segment()
    return render_template('index.html', list=list)


#TODO: Figure out how to create global variables for input_variable and dictionaries
# for layer names. Alternative is to scrape from folders and create limits on what can
# be uploaded and ways to clear the folders from GUI
#TODO: Figure out importing functions from different files. May involve init.py
# dictionary of layers to uploads
#TODO: need permanence, secondary inputs refresh after each upload. This 
# can be solved if we figure out global variables
#Use this?:
#  https://stackoverflow.com/questions/28423069/store-large-data-or-a-service-connection-per-flask-session/28426819#28426819
# dictionary of layer numbers to layer names
# pass list of video, image, or nothing, for each layer. Default is nothing.