import os
from flask import Flask, render_template, flash, request, redirect
from werkzeug.utils import secure_filename
from processing import open_file, pre_layer_replace
from layerReplacement.read_labels import read_labels

UPLOAD_FOLDER = os.getcwd()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif',
                      'mp4'}  # additional extensions, .mp4, .mov, .tiff, .avi
primary_input = None
have_segmented = False
layer_dict = {}
layer_list = []


# Takes in a filename and verifies whether it is in the bounds of the allowed
# extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Uploads the primary input
def upload_primary_input():
    if request.method == 'POST':
        if 'file' in request.files:
            global primary_input
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


def upload_secondary_input():
    if request.method == 'POST':
        if 'secondary_input' in request.files:
            secondary_input = request.files['secondary_input']
            curr_layer = request.form['index']
            if secondary_input.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if secondary_input and allowed_file(secondary_input.filename):
                filename = secure_filename(secondary_input.filename)
                new_path = os.path.join(app.config['UPLOAD_FOLDER'], "static/data/secondary_inputs", filename)
                secondary_input.save(new_path)
                print("curr layer", curr_layer)
                layer_dict[curr_layer].append(request.form["input_type"])
                layer_dict[curr_layer].append(new_path)
                print("secondary layer dict", layer_dict)
                return new_path
    return


def do_segment():
    global counter
    if primary_input:
        if request.method == 'POST':
            if 'segment_button' in request.form:

                if request.form['segment_button'] == 'Segment':
                    unique_layers = open_file(primary_input)
                    number_to_label = read_labels("./layerReplacement/labels.json")
                    global layer_list
                    layer_list = []
                    for layer_number in unique_layers:
                        layer_list.append(number_to_label[layer_number])

                    global layer_dict
                    layer_dict = {k: [] for k in layer_list}
                    global have_segmented
                    have_segmented = True
                    print("segment layer dict", layer_dict)
                    return layer_list
    else:
        # no primary input, display message about uploading primary input?
        return


def replace_layers():
    if request.method == 'POST':
        if 'replace_button' in request.form:
            if request.form['replace_button'] == 'Replace Layers':
                print("replaceLAYERS is occuring")
                pre_layer_replace(layer_dict) # giving it layer dict
                # TODO: add replace layer algorithm here. This will be called when the replace layer
                # button is pressed. Layers and their associated secondary_inputs are stored in layer_dict
                return
    return


@app.route('/', methods=['GET', 'POST'])
# Runs the upload method and returns the rendered page
def index():
    upload_primary_input()
    upload_secondary_input()
    print("primary", primary_input)
    do_segment()
    return render_template('index.html', layer_list=layer_list, have_segmented=have_segmented)

# TODO: Clear data folders at the beginning of each run?
# TODO: flashes interrupt flow of gui, change these to messages?
