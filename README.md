# X/Twitter Misinformation Classifier

Uses Logistic Regression to classify tweets as misinformation or not based on a large dataset of labeled tweets.

### Stack:
- Python (Pandas, Scikit-learn, numpy, Flask)
- React (Frontend)

### Setup:
#### .env files:

ml_model/.env
```
BEARER_TOKEN=Twitter API Bearer Token (Not necessary)
DB_NAME=Postgres DB name
DB_USER=Postgres DB user
DB_PASSWORD=Postgres DB password
DB_HOST=Postgres DB host
DB_PORT=Postgres DB port
```

frontend/.env
```
VITE_BACKEND_URL=backend url (localhost is fine)
```

#### Running:
frontend:
```
cd frontend
npm i
npm run dev
```

backend:
```
pip install -r requirements.txt
cd ml_model
python app.py
```

### Credits:

---
Dataset: 
From: The Largest Social Media Ground-Truth Dataset for Real/Fake Content: TruthSeeker
By: Sajjad Dadkhah; Xichen Zhang; Alexander Gerald Weismann; Amir Firouzi; Ali A. Ghorbani
DOI: 10.1109/TCSS.2023.3322303
