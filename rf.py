import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------
# Step 1: Create Dataset
# --------------------------
data = {
    'Hours_Studied': [2, 4, 5, 7, 8, 1, 3, 6, 9, 10],
    'Attendance':    [60, 70, 75, 80, 85, 55, 65, 78, 90, 95],
    'Result':        ['Fail', 'Fail', 'Pass', 'Pass', 'Pass', 
                      'Fail', 'Fail', 'Pass', 'Pass', 'Pass']
}

df = pd.DataFrame(data)

# --------------------------
# Step 2: Encode Target
# --------------------------
le = LabelEncoder()
df['Result'] = le.fit_transform(df['Result'])  # Fail=0, Pass=1

# Features & Target
X = df[['Hours_Studied', 'Attendance']]
y = df['Result']

# --------------------------
# Step 3: Split Data
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------
# Step 4: Train Random Forest
# --------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# --------------------------
# Step 5: Make Predictions
# --------------------------
y_pred = rf.predict(X_test)

# --------------------------
# Step 6: Evaluate Model
# --------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
