from flask import Flask, render_template, Response ,request
from model import CovidTest
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
# @app.route('/r')
# def r():
#     return render_template('result.html',res="Covid Negative")
@app.route('/result',methods=['GET','POST'])
def result():
    if request.method=='POST':
        img=request.files['img']
        img.save('static\input_img_db/'+img.filename)
        model = CovidTest("model1.json", "model_weights1.h5")
        img=cv2.imread('static\input_img_db/'+img.filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img=img.reshape(-1, 299, 299, 1)
        res=model.predict(img)
        return render_template('result.html',res=res)

if __name__ == '__main__':
    app.run(debug=True)
