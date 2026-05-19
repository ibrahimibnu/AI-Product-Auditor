import pandas as pd
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading the dataset...")

df=pd.read_csv("ecommerce_product_performance.csv")
df

print("\nInspecting missing values and datatypes...")
print("Missing values per column: ")
print(df.isna().sum())
print("\nData types: ")
print(df.dtypes)

#Cleaning missing values before splitting 

df_clean = df.dropna(subset=['Product_Price', 'Product_Rating', 'Category_ID'])
print(f"\n✅ Cleaned data! Kept {len(df_clean)} aligned rows where all features exist.")



print("\nSeperating Features (X) and Target (y)...")

#X is our inputs (Rating/Condition and Category ID)
X=df_clean[['Product_Rating','Category_ID']]

#y is what we want to predict (Price)
y = df_clean['Product_Price']

print("\nApplying Feature Scaling...")

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
print("Sample scaled features:\n",X_scaled[:3])

print("\nSplitting into Training and Testing sets... ")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)
print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

print("\nTraining the Regression Model...")
model = LinearRegression()
model.fit(X_train, y_train)

print("Making predictions on test data...")
y_pred = model.predict(X_test)
print("Sample predictions:\n",y_pred[:5])

print("\nEvaluating Model Performance...")
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"R² Score (Accuracy/Fit): {r2:.4f}")

print("\nSaving the Model and Scaler for the UI App... ")
#Save the trained model
with open("models/pricing_model.pkl","wb") as f:
    pickle.dump(model, f)

#Save the scalar so the app can scale user inputs exactly the same way
with open("models/scaler.pkl","wb") as f:
    pickle.dump(scaler, f)

print("Success! Professional pipeline executed completely.")        