import pandas as pd

# Assuming corrected file paths
ell_df_path = '../Spring 2024 Conversation Partners Request Form (ELL).csv'
volunteer_df_path = '../Spring 2024 Volunteer Form.csv'
matches_df_path = 'Spring 2024 Conversation Partners - Matches (1).csv'

# Loading datasets
ell_df = pd.read_csv(ell_df_path)
volunteer_df = pd.read_csv(volunteer_df_path)
matches_df = pd.read_csv(matches_df_path)

# feature extraction for ELL participants
ell_features = ell_df[[
    'First Name', 'Last Name', 'TAMU Email',  # Basic identification information
    'Major',
    'Classification',
    'Please select your preference for meetings with your conversation partner.',
]].copy()

# feature extraction for Volunteer participants (similar approach as for ELL participants)
volunteer_features = volunteer_df[[
    'First Name', 'Last Name', 'TAMU Email',
    'Major',
    'Classification',
    'If you are interested in a Conversation Partnership, please select your preference for meeting.',
]].copy()

# Preparing the Matches dataset for labeling
matches_simplified = matches_df[['NNS - UIN', 'NS - UIN']].copy()
matches_simplified['Matched'] = 1

# Displaying the first few rows of each DataFrame to verify the feature selection
print("ELL Features Preview:", ell_features.head(), sep='\n')
print("\nVolunteer Features Preview:", volunteer_features.head(), sep='\n')
print("\nMatches Simplified Preview:", matches_simplified.head(), sep='\n')


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identifying categorical columns for preprocessing
categorical_cols_ell = ['Major', 'Please select your preference for meetings with your conversation partner.']
categorical_cols_volunteer = ['Major', 'If you are interested in a Conversation Partnership, please select your preference for meeting.']

# Setup preprocessing for categorical columns with OneHotEncoder and SimpleImputer
categorical_imputer = SimpleImputer(strategy='most_frequent')
categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# Apply preprocessing for ELL dataset
preprocessor_ell = ColumnTransformer(
    transformers=[
        ('cat', Pipeline(steps=[
            ('imputer', categorical_imputer),
            ('encoder', categorical_encoder)]), categorical_cols_ell)
    ],
    remainder='passthrough'
)

ell_features_preprocessed = preprocessor_ell.fit_transform(ell_df[categorical_cols_ell])

# Apply preprocessing for Volunteer dataset
preprocessor_volunteer = ColumnTransformer(
    transformers=[
        ('cat', Pipeline(steps=[
            ('imputer', categorical_imputer),
            ('encoder', categorical_encoder)]), categorical_cols_volunteer)
    ],
    remainder='passthrough'
)

volunteer_features_preprocessed = preprocessor_volunteer.fit_transform(volunteer_df[categorical_cols_volunteer])

print("Data Cleaning and Preprocessing Complete.")
print("ELL Features Preprocessed Shape:", ell_features_preprocessed.shape)
print("Volunteer Features Preprocessed Shape:", volunteer_features_preprocessed.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Simulate a scenario where we're only using ELL data for demonstration
features = ell_features_preprocessed

labels = np.random.randint(2, size=features.shape[0])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

# Initializing the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from joblib import dump

# Example: Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate with additional metrics
print("ROC AUC Score:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))
print("Best Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Deployment preparation: Serialize the model
model_filename = 'final_model.joblib'
dump(best_model, model_filename)
print(f"Model saved as {model_filename}")
