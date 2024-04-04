import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


# Load the Iris dataset
iris_df = pd.read_csv('iris.data', names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'])
print(iris_df)
iris_df.head()
print(iris_df.head())

# statistical analysis about the data
iris_df.describe()
print(iris_df.describe())

# Explore the dataset
sns.pairplot(iris_df, hue='Species')
plt.show()

# Prepare data for analysis
X = iris_df.drop('Species', axis=1)
y = iris_df['Species']



# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Example new data points
new_data =np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])

# Predict species for new data
new_predictions = svm_model.predict(new_data)
print("Predicted Species: {}".format(new_predictions))

# Save and load the model
with open('Iris_SVM_Model.pickle', 'wb') as f:
    pickle.dump(svm_model, f)

with open('Iris_SVM_Model.pickle', 'rb') as f:
    loaded_model = pickle.load(f)

loaded_predictions = loaded_model.predict(new_data)
print("Loaded Model Predictions:", loaded_predictions)
