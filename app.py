# Importing essential libraries
from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf


app = Flask(__name__) 
app.secret_key = 'O.\x89\xcc\xa0>\x96\xf7\x871\xa2\xe6\x9a\xe4\x14\x91\x0e\xe5)\xd9'

# Load the Random Forest CLassifier model

filename1 = 'Models/cancer-model.pkl'
classifier = pickle.load(open(filename1, 'rb'))
rf = pickle.load(open(filename1, 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/cancer')
def cancer():
    return render_template('cancer.html')


@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    if request.method == 'POST':
        rad = float(request.form['Radius_mean'])
        tex = float(request.form['Texture_mean'])
        par = float(request.form['Perimeter_mean'])
        area = float(request.form['Area_mean'])
        smooth = float(request.form['Smoothness_mean'])
        compact = float(request.form['Compactness_mean'])
        con = float(request.form['Concavity_mean'])
        concave = float(request.form['concave points_mean'])
        sym = float(request.form['symmetry_mean'])
        frac = float(request.form['fractal_dimension_mean'])
        rad_se = float(request.form['radius_se'])
        tex_se = float(request.form['texture_se'])
        par_se = float(request.form['perimeter_se'])
        area_se = float(request.form['area_se'])
        smooth_se = float(request.form['smoothness_se'])
        compact_se = float(request.form['compactness_se'])
        con_se = float(request.form['concavity_se'])
        concave_se = float(request.form['concave points_se'])
        sym_se = float(request.form['symmetry_se'])
        frac_se = float(request.form['fractal_dimension_se'])
        rad_worst = float(request.form['radius_worst'])
        tex_worst = float(request.form['texture_worst'])
        par_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        smooth_worst = float(request.form['smoothness_worst'])
        compact_worst = float(request.form['compactness_worst'])
        con_worst = float(request.form['concavity_worst'])
        concave_worst = float(request.form['concave points_worst'])
        sym_worst = float(request.form['symmetry_worst'])
        frac_worst = float(request.form['fractal_dimension_worst'])

        data = np.array([[rad, tex, par, area, smooth, compact, con, concave, sym, frac, rad_se, tex_se, par_se, area_se, smooth_se, compact_se, con_se, concave_se,
                          sym_se, frac_se, rad_worst, tex_worst, par_worst, area_worst, smooth_worst, compact_worst, con_worst, concave_worst, sym_worst, frac_worst]])
        my_prediction = rf.predict(data)

        return render_template('c_result.html', prediction=my_prediction)




# this function use to predict the output for Fetal Health from given data
def fetal_health_value_predictor(data):
    try:
        # after get the data from html form then we collect the values and
        # converts into 2D numpy array for prediction
        data = list(data.values())
        data = list(map(float, data))
        data = np.array(data).reshape(1, -1)
        # load the saved pre-trained model for new prediction
        model_path = 'Models/fetal-health-model.pkl'
        model = pickle.load(open(model_path, 'rb'))
        result = model.predict(data)
        result = int(result[0])
        status = True
        # returns the predicted output value
        return (result, status)
    except Exception as e:
        result = str(e)
        status = False
        return (result, status)


# this route for prediction of Fetal Health
@app.route('/fetal_health', methods=['GET', 'POST'])
def fetal_health_prediction():
    if request.method == 'POST':
        # geting the form data by POST method
        data = request.form.to_dict()
        # passing form data to castome predict method to get the result
        result, status = fetal_health_value_predictor(data)
        if status:
            # if prediction happens successfully status=True and then pass uotput to html page
            return render_template('fetal_health.html', result=result)
        else:
            # if any error occured during prediction then the error msg will be displayed
            return f'<h2>Error : {result}</h2>'

    # if the user send a GET request to '/fetal_health' route then we just render the html page
    # which contains a form for prediction
    return render_template('fetal_health.html', result=None)


@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('pneumonia.html', message = message)
    return render_template('pneumonia_predict.html', pred = pred)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')


if __name__ == '__main__':
    app.run(debug=True)
