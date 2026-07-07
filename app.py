from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import ssl

app = Flask(__name__)

ssl._create_default_https_context = ssl._create_unverified_context

url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
df = pd.read_csv(url, sep='\t', on_bad_lines='skip')
df = df.dropna()

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['document'])
y = df['label']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("모델 학습 완료!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    review = request.form['review']
    review_vec = vectorizer.transform([review])
    prediction = model.predict(review_vec)[0]
    result = "긍정 😊" if prediction == 1 else "부정 😢"
    return render_template('index.html', result=result, review=review)

if __name__ == '__main__':
    import os
port = int(os.environ.get('PORT', 5001))
app.run(host='0.0.0.0', port=port)
