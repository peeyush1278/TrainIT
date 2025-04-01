import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
# warnings.filterwarnings('ignore')

# Read and prepare the data
df = pd.read_csv("Rainfall_Data_LL.csv")
df.drop(columns=["Name"], inplace=True)

# Prepare features for subdivision prediction
features = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 
           'ANNUAL', 'Jan-Feb', 'Mar-May', 'June-September', 'Oct-Dec', 'Latitude', 'Longitude']

for i in range(1,len(features)):
    X = df[features[:i]]
    y = df['SUBDIVISION']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)
    ada_params = {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'algorithm': ['SAMME']
    }

    ada = AdaBoostClassifier(estimator=base_estimator, random_state=42)
    grid_search = GridSearchCV(ada, ada_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train_scaled, y_train)

    best_ada = grid_search.best_estimator_
    y_pred = best_ada.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nFeatures used: {features[:i]}")
    print(f"Test Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - AdaBoost (Features: {len(features[:i])})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    # Save the cleaned dataset
    df.to_csv('Rainfall_Data_Clean.csv', index=False)

    print("\nFeatures available for subdivision prediction:", features)
    print("\nSample of the prepared data:")
    print(df[['SUBDIVISION'] + features].head())


