from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open('spotify_model.pkl', 'rb') as f:
    model = pickle.load(f)

features = ['danceability', 'energy', 'valence', 'tempo', 
            'acousticness', 'instrumentalness', 'speechiness', 'liveness']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    input_values = [
        float(data['danceability']),
        float(data['energy']),
        float(data['valence']),
        float(data['tempo']),
        float(data['acousticness']),
        float(data['instrumentalness']),
        float(data['speechiness']),
        float(data['liveness'])
    ]
    
    X = pd.DataFrame([input_values], columns=features)
    prediction = model.predict(X)[0]
    
    result = "🔥 THIS WILL BE A HIT!" if prediction == 1 else "❌ Not Popular"
    
    return jsonify({'prediction': result, 'is_popular': int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)