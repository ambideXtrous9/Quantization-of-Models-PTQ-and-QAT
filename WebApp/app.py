import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
from flask import Flask, request, render_template, redirect, url_for,send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from QATpred import modelpred


# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def delete_old_files(folder):
    """Delete all files in the specified folder."""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # Delete old files before uploading a new one
        delete_old_files(app.config['UPLOAD_FOLDER'])


        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image = Image.open(file)
            image = image.resize((400, 400))
            image.save(filepath)

            # Start the timer
            start_time = time.time()
            
            confidence, predicted_class = modelpred(filepath)

            # Map predicted class to label
            class_names = {0: 'Cat', 1: 'Dog'}
            predicted_label = class_names[predicted_class.item()]

            # End the timer
            end_time = time.time()
            execution_time = end_time - start_time

            return render_template('result.html', 
                                   filename=filename, 
                                   predicted_class=predicted_label, 
                                   confidence=confidence.item(),
                                   execution_time=execution_time)
        
    return render_template('index.html')

@app.route('/uploads/<filename>')

def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
