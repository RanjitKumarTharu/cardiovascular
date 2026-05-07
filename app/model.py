import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('data/Cardiovascular_Disease.csv')

df['age'] = round(df['age'] / 365, 2)

df = df[(df['ap_hi'] >= 120) & (df['ap_hi'] <= 210)]
df = df[(df['ap_lo'] >= 50) & (df['ap_lo'] <= 100)]

MODEL_PATH = 'model/cardio_model.pkl'
SCALER_PATH = 'model/cardio_scaler.pkl'

def cardio_predict():
    features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol',
                'gluc', 'smoke', 'alco', 'active']
    target = 'cardio'

    X = df[features]
    Y = df[target]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    scaler = StandardScaler()  # (Xi-mean / sd)
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)

    model = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train_scale, Y_train)
    model.predict(X_test_scale)

    joblib.dump(model,MODEL_PATH)
    joblib.dump(scaler,SCALER_PATH)


    return model , scaler

def load_model_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model ,scaler