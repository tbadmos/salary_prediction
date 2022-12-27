from flask import Flask, render_template, request

import sys
#sys.path.append('/mnt/c/users/olatu/documents/ASQsixsigma/DSDJ/Module4ProjectPortfolio/SalaryPredictionProject/salarypredictionportfolio/modules')
#sys.path.append('./modules')
import helper_web_app as hlp

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])


def predict():
    to_predict_params = request.form.to_dict()
    
    prediction = hlp.predict(to_predict_params)

    return render_template('predict.html', to_predict_params = to_predict_params, prediction = prediction)
   # return to_predict_list

if __name__ == "__main__":
    app.run(debug=True)