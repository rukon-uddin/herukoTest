from flask import Flask, render_template, request, url_for, redirect, flash
from flask.wrappers import Request
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.image as mpimg
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def hello_world():

    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)


        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print('upload_image filename: ' + filename)
            model = tf.keras.models.load_model("E:/FlaskTutorial/app/A-unet_model.h5")
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            x = mpimg.imread(path)
            original_image = x
            h,w,_ = x.shape
            x = np.resize(x, (256,256,3))
            x = x / 255.0
            x = x.astype(np.float32)
            x = np.expand_dims(x, axis=0)

            pred_mask = model.predict(x)
            pred_mask = pred_mask[0]

            # pred_mask = np.concatenate(
            #     [
            #         pred_mask, pred_mask, pred_mask
            #     ], axis=2
            # )

            pred_mask[pred_mask > 0.5] = 255
            pred_mask = pred_mask.astype(np.uint8)
            pred_mask = np.resize(pred_mask, (w, h))
            pred_mask = Image.fromarray(pred_mask).convert('RGB')
            pred_mask.save("static/uploads/mask.png")
            return render_template('index.html', filename='mask.png')
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.debug = True
    app.run()

