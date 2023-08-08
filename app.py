from flask import Flask,request,render_template
import numpy as np
import cv2

import pickle

model=pickle.load(open('model.pkl','rb'))


app=Flask(__name__)

@app.route('/')
def inp():
    return render_template("input.html")

@app.route('/pred',methods=['post'])
def pred():
    imag=request.files['myfile']
    img=cv2.imdecode(np.fromstring(imag.read(),np.uint8),cv2.IMREAD_UNCHANGED)[:,:,::-1]
    img=img/255
    img = cv2.convertScaleAbs(img)
    resized_img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    reshaped_img = gray_img.reshape((28, 28, 1))
    final_pred=np.argmax(model.predict(reshaped_img.reshape(1,28,28,1)))
    
    return render_template('input.html', data=str(final_pred))
    
    
    
if __name__=='__main__':
    app.run(host='localhost',port=5400)


