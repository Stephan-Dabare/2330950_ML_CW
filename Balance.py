from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('bank-full.csv', sep=';')

# Separate the features and the target
X = data.drop('y', axis=1)
y = data['y']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Counting the number of instances in each class before oversampling
counter = Counter(y_train)
total_before = sum(counter.values())
print('Before:', counter)
for cls, count in counter.items():
    print(f'Class {cls}: {count} ({count / total_before:.2%})')
plt.bar(counter.keys(), counter.values())
plt.title('Class Distribution Before Oversampling')
plt.show()

# Oversampling the train dataset using SMOTE
smt = SMOTE(random_state=139)
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

# Counting the number of instances in each class after oversampling
counter = Counter(y_train_sm)
total_after = sum(counter.values())
print('After:', counter)
for cls, count in counter.items():
    print(f'Class {cls}: {count} ({count / total_after:.2%})')
plt.bar(counter.keys(), counter.values())
plt.title('Class Distribution After Oversampling')
plt.show()