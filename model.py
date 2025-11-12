import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Step 1: Load dataset
df = pd.read_csv('housing.csv')

# Step 2: Encode location
le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])

# Step 3: Split features & label
X = df[['area', 'bedrooms', 'age', 'location']]
y = df['price']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Save model & encoder
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(le, open('encoder.pkl', 'wb'))

print("✅ Model trained successfully and saved as model.pkl & encoder.pkl")
