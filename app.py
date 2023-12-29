# dependencies
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import cv2
from flask import Flask, render_template, request
import time
import os
from PIL import Image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # format gambar yang dibutuhkan
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'  # folder tempat untuk upload
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # size maksimal image

# fungsi yang menampilkan hasil prediksi


def result(model, run_time, probs, img):
    class_list = {'Paper': 0, 'Rock': 1, 'Scissors': 2}
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys())
    return render_template('/result.html', labels=labels,
                           probs=probs, model=model, pred=idx_pred,
                           run_time=run_time, img=img)


@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# route utama yang menampilkan index.html


@app.route('/')
def index():
    return render_template('/index.html')

# route untuk memproses dan memprediksi image


@app.route('/predict', methods=['POST'])
def predict():
    # load model resnet yang sudah di train
    model = load_model('./model/model_rps.h5')
    file = request.files["file"]  # mengambil gambar dari input di html

    # disimpan menjadi gambar sementara
    file.save(os.path.join('static', 'temp.jpg'))
    img = cv2.cvtColor(np.array(Image.open(file)),
                       cv2.COLOR_BGR2RGB)  # Convert BGR ke RGB
    # resize menjadi gambar yang sesuai dengan input model yang digunakan dan di rescale
    img = np.expand_dims(cv2.resize(
        img, (224, 224)).astype('float32') / 255, axis=0)

    start = time.time()
    pred = model.predict(img)[0]
    labels = (pred > 0.5).astype(int)
    # menghitung waktu total runtime untuk model memprediksi gambar
    runtimes = round(time.time()-start, 4)
    respon_model = [round(elem * 100, 2) for elem in pred]
    return result('Model Rock Paper Scissors', runtimes, respon_model, 'temp.jpg')


# akan dijalankan pertama kali saat menjalankan server
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=2000)
