from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np

app=Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('sc.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
     features = [float(x) for x in request.form.values()]
     final_features = [np.array(features)]
     final_scaled_features = scaler.transform(final_features)
     prediction = model.predict(final_scaled_features)
     output = 'Potable' if prediction[0] == 1 else 'Not Potable'
    except Exception as e:
     output = "Error: " + str(e)


    return render_template('index.html', prediction_text='Prediction: {}'.format(output))


if __name__=='__main__':
    app.run(debug=True)