from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__)

# 네이버 영화 리뷰 데이터 불러오기
# → 탭으로 구분된 파일이라 sep='\t' 써요
# → 오류나는 행은 건너뛰어요
df = pd.read_csv("ratings_train.txt", sep='\t', on_bad_lines='skip')

# 결측값 제거
# → 비어있는 리뷰 제거해요
df = df.dropna()

# 모델 학습
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
    app.run(debug=True, port=5001)