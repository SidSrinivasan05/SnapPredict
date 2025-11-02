import pandas as pd
import numpy as np
from pprint import pprint

# import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/nfl_pbp_2019_2022.csv')

pprint( df.head() )
print("___" * 50)
pprint( list( df.columns ) )

df_pass = df[ ['complete_pass', 'ydstogo', 'yardline_100', 'quarter_seconds_remaining'] + ['down', 'play_type', 'shotgun', 'posteam', 'defteam']]

df_pass = df_pass[df_pass['complete_pass'].notna()].reset_index(drop=True)


pprint( df_pass.head() )


print("___" * 50)

def test(df_pass):
    target = 'complete_pass'
    numeric_features = ['ydstogo', 'yardline_100', 'quarter_seconds_remaining']
    categorical_features = ['down', 'play_type', 'shotgun', 'posteam', 'defteam']

    X = df_pass[numeric_features + categorical_features]
    y = df_pass[target]

    # 3️⃣ Handle missing values
    X[numeric_features] = X[numeric_features].fillna(0)
    X[categorical_features] = X[categorical_features].fillna('missing')

    # 3️⃣5️⃣ Convert all categorical columns to strings
    X[categorical_features] = X[categorical_features].astype(str)

    # 4️⃣ Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5️⃣ Preprocessing
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # 6️⃣ Model pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    clf = Pipeline([
        ('pre', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 7️⃣ Fit
    clf.fit(X_train, y_train)

    # 8️⃣ Predict
    from sklearn.metrics import accuracy_score
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    
# df_pass.to_csv('data/cleaned/processed_nfl_pass_data.csv', index=False)

print(len(list(df.columns) ))