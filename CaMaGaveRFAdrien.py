import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the provided Excel files
classif_251_347 = pd.read_excel('classif 251-347.xlsx')
classif_test_v2 = pd.read_excel('classif_test_v2.xlsx')
donnees2_v2 = pd.read_excel('données2_v2.xlsx')

# Drop rows with NaN in the 'ID' column in classif_251_347
classif_251_347 = classif_251_347.dropna(subset=['ID'])

# Prepare the training data
features = classif_test_v2.drop(columns=['ID', 'bug type', 'species'])
labels = classif_test_v2['bug type']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Impute missing values with the mean of each column
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_imputed, y_train)

# Test the model
y_pred = rf_classifier.predict(X_test_imputed)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Prepare the data for prediction
features_to_predict = donnees2_v2.drop(columns=['ID'])  # Assuming 'ID' column is not a feature

# Impute missing values in the data to predict
features_to_predict_imputed = imputer.transform(features_to_predict)

# Predict the bug type
predictions = rf_classifier.predict(features_to_predict_imputed)

# Add the predictions to the original dataframe
donnees2_v2['Predicted Bug Type'] = predictions

# Save the results to a new Excel file
output_file = "predictions.xlsx"
donnees2_v2.to_excel(output_file, index=False)

# Merge with classif_251_347 to add the correct answers
merged_results = donnees2_v2.merge(classif_251_347, on='ID', how='left')

# Calculate the accuracy if ground truth is available in donnees2_v2
if 'Bug Type' in merged_results.columns:
    true_labels = merged_results['Bug Type']
    # Filter out rows with NaN values in true_labels
    valid_indices = ~true_labels.isna()
    filtered_true_labels = true_labels[valid_indices]
    filtered_predictions = predictions[valid_indices]
    # Calculate the accuracy with filtered values
    accuracy_on_donnees2 = accuracy_score(filtered_true_labels, filtered_predictions)
    # Calculate the percentage of correct predictions
    correct_predictions = (filtered_true_labels == filtered_predictions).sum()
    total_predictions = len(filtered_true_labels)
    percentage_correct = (correct_predictions / total_predictions) * 100
else:
    accuracy_on_donnees2 = None
    percentage_correct = None

# Display the merged results
print("Output file saved as:", output_file)
print("Accuracy on provided data:", accuracy_on_donnees2)
print("Percentage of correct predictions:", percentage_correct)
print("Merged Results:\n", merged_results.loc[valid_indices, ['ID', 'Predicted Bug Type', 'Bug Type']].to_string(index=False))
