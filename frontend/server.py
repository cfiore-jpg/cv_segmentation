import os
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.getcwd()
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Takes in a filename and verifies whether it is in the bounds of the allowed
# extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def download_file(name):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], name)

# Uploads the selected file to static/data/uploads
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            new_path = os.path.join(app.config['UPLOAD_FOLDER'], "static/data/uploads", filename)
            file.save(new_path)
            #url_path = os.path.join("data/uploads", filename)
            #return redirect(url_for('download_file', name=url_path))
            return "File uploaded to static/data/uploads"
    return

@app.route('/', methods=['GET', 'POST'])
# Runs the upload method and returns the rendered page
def index():
    upload_return = upload_file()
    return render_template('index.html', upload_return=upload_return)

#2.) Displaying list of layers with options for secondary input



