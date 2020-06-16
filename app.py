import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from test import return_test_sample
from multi_train import train_fcn
import os 
import logging as logger
# from gevent.pywsgi import WSGIServer
logger.basicConfig(level='DEBUG')

# Creating the app
app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

# Loading the best model after running the multi_train
files = os.listdir('./mlruns/0/')
r2_values = []
file_names = []
for dir_id in files:
    if dir_id != 'meta.yaml':
        file_names.append(dir_id)
        path = './mlruns/0/'+f'{dir_id}'+'/metrics/r2'
        fp = open(path)
        data = fp.read()
        r2_values.append(round(float(data.split(' ')[1]), 5))
        
# Getting the best model
idx = np.argmax(r2_values)
# print(r2_values)
# print(idx)
# print(file_names)
model_dir = os.listdir('./mlruns/0/'+file_names[idx]+'/artifacts/')
model_path = './mlruns/0/'+file_names[idx]+'/artifacts/'+model_dir[1]+'/model.pkl'
# print(model_path)
model = pickle.load(open(model_path, 'rb'))

# Getting the test data samples
X_test, test_data = return_test_sample()
X_test = np.round(X_test)

@app.route('/')
def home():
    '''
    For rendering home page
    '''
    test_list = []
    for i,k in enumerate(X_test):
        value = list(k)
        test_list.append({f"Sample": i, f"values": value})

    return render_template('index.html', test_list = test_list, best_model=model_dir[1])

@app.route('/predict/<idx>')
def predict(idx):
    '''
    For rendering results on HTML GUI
    '''
    # Getting predictions
    prediction = model.predict(test_data[int(idx)].reshape(1, -1))

    # Rounding off
    output = prediction[0]*1000
    output = round(output, 3)

    # Displaying the other samples
    test_list = []
    for i,k in enumerate(X_test):
        value = list(k)
        test_list.append({f"Sample": i, f"values": value})

    return render_template('results.html',  test_data = X_test[int(idx)], test_list = test_list,
                            prediction_text=f'Predicted House price : $ {output}', best_model=model_dir[1])

@app.route('/retrain', methods=['POST', 'GET'])
def retrain():
    '''
    For retraining the model
    '''
    r2 = None
    if request.method == 'POST':
    
        inputs = list(request.form.values())
        print(inputs)
        print(type(inputs))
            
        if inputs[0] == 'Linear Regression':
            # Calling the function for retraining
            r2 = train_fcn(exp_name='Compare_algo', model_name=inputs[0], penalty=None, alpha=None, 
                            l1_ratio=None, learning_rate=None, n_estimators=None, max_depth=None)
            

        elif inputs[0] == 'SGD Regression':
            # Calling the function for retraining
            r2 = train_fcn(exp_name='Compare_algo', model_name=inputs[0], penalty=inputs[1], 
                            alpha=float(inputs[2]), l1_ratio=float(inputs[3]), 
                            learning_rate=inputs[4], n_estimators=None, max_depth=None)

        elif inputs[0] == 'RF Regression':
            # Calling the function for retraining
            r2 = train_fcn(exp_name='Compare_algo', model_name=inputs[0], penalty=None, 
                            alpha=None, l1_ratio=None, learning_rate=None, 
                            n_estimators=int(inputs[1]), max_depth=int(inputs[2]))

        elif inputs[0] == 'DT Regression':
            # Calling the function for retraining
            r2 = train_fcn(exp_name='Compare_algo', model_name=inputs[0], penalty=None, 
                            alpha=None, l1_ratio=None, learning_rate=None, 
                            n_estimators=int(inputs[1]), max_depth=int(inputs[2]))

        elif inputs[0] == 'GBDT Regression':
            # Calling the function for retraining
            r2 = train_fcn(exp_name='Compare_algo', model_name=inputs[0], penalty=None, 
                            alpha=None, l1_ratio=None, learning_rate=float(inputs[3]), 
                            n_estimators=int(inputs[1]), max_depth=int(inputs[2]))

        elif inputs[0] == 'XGB Regression':
            # Calling the function for retraining
            r2 = train_fcn(exp_name='Compare_algo', model_name=inputs[0], penalty=None, 
                            alpha=None, l1_ratio=None, learning_rate=float(inputs[3]), 
                            n_estimators=int(inputs[1]), max_depth=int(inputs[2]))

    # text=f'The R_sqaure value after training is {r2}'


    return render_template('retrain.html', text=f'The R_sqaure value after training is {r2}', best_model=model_dir[1])


# if __name__ == "__main__":

    # # Debug/Development
    # logger.debug("Starting Flask Server")
    # app.run(host='0.0.0.0', port='8200' ,debug=True)

    # Production
    # Keep WSGIServer(('', 8200), app) while prduction
#    logger.debug("Server running at http://127.0.0.1:8200/")
#    http_server = WSGIServer(('', 8200), app)
#    http_server.serve_forever()
