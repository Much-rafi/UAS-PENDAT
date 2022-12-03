import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from flask import Flask
import os
app = Flask(__name__)
# model = pickle.load(open("model.pkl", "rb"))


@app.route('/', methods=['GET', 'POST'])
# @app.route('/hasil.html', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        gravity = float(request.form['gravity'])
        ph = float(request.form['ph'])
        osmo = float(request.form['osmo'])
        cond = float(request.form['cond'])
        urea = float(request.form['urea'])
        calc= float(request.form['calc'])
        metod=request.form['metod']
        
        val = np.array([gravity, ph, osmo, cond,urea,calc])
        datain=[np.array(val)]

        scalar_path= os.path.join('models','scalerData')
        scalar = pickle.load(open(scalar_path, 'rb'))
        

        final_features = scalar.transform(datain)
        
        if metod == "bayes":

            model_path = os.path.join('models','modelGinjalNB.sav')
            model = pickle.load(open(model_path, 'rb'))
            res = model.predict(final_features)
            if res == 0:
                outdata='Tidak Terdeteksi'
            if res == 1:
                outdata='Terdeteksi'  
            return render_template('hasil.html', prediction_text=outdata)
        if metod == "pohon":

            model_path = os.path.join('models','modelGinjalDT.sav')
            model = pickle.load(open(model_path, 'rb'))
            res = model.predict(final_features)
            if res == 0:
                outdata='Tidak Terdeteksi'
            if res == 1:
                outdata='Terdeteksi'  
            return render_template('hasil2.html', prediction_text=outdata)    
        else:
           
            model_path = os.path.join('models','modelGinjalRF.sav')
            model = pickle.load(open(model_path, 'rb'))
            res = model.predict(final_features)
            if res == 0:
                outdata='Tidak Terdeteksi'
            if res == 1:
                outdata='Terdeteksi'  
        return render_template('hasil3.html', prediction_text=outdata)    
    return render_template('index.html')


# if __name__ == "__main__":
#     app.run(debug=True)
if __name__=="__main__":
    app.run(debug=True)