# 🎬 영화 리뷰 감성 분석 웹 서비스

## 프로젝트 소개
영화 리뷰를 입력하면
AI가 긍정/부정을 분석해주는 웹 서비스예요!
한국어, 영어 둘 다 지원해요!

## 사용 기술
- Python
- Flask
- Scikit-learn
- TF-IDF Vectorizer
- HTML/CSS

## 학습 데이터
- 네이버 영화 리뷰 20만개 (한국어)
- IMDB 영화 리뷰 5만개 (영어)

## 실행 방법
```bash
pip3 install flask scikit-learn pandas
python3 app.py
```
브라우저에서 http://127.0.0.1:5001 접속!

## 결과 예시
- "너무 재밌게 봤어요!" → 긍정 😊
- "너무 별로였어요" → 부정 😢
