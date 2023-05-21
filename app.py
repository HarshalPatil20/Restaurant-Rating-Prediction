import numpy as np
from flask import Flask , request, render_template
import joblib


app= Flask(__name__)
##load the model
#model = pickle.load(open('model.pkl','rb'))
model = joblib.load('extratree_regressor.joblib')

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],1)

    return render_template('index.html',prediction_text='Restaurant Rating is: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)