# ================= STEP 1: Set Kaggle API Credentials =================
import os

# 🔐 Replace with your actual Kaggle username and API key
os.environ['KAGGLE_USERNAME'] = 'rajchakravarti'
os.environ['KAGGLE_KEY'] = '33f33a36de39838d9640e1a2260ab6f0'

# ================= STEP 2: Install & Download Dataset =================
!pip install -q kaggle

# Download dataset
!kaggle datasets download -d olistbr/brazilian-ecommerce

# Unzip the dataset
!unzip -q brazilian-ecommerce.zip

# ================= STEP 3: Import Libraries =================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================= STEP 4: Load and Merge Data =================
orders = pd.read_csv("olist_orders_dataset.csv")
items = pd.read_csv("olist_order_items_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")
categories = pd.read_csv("product_category_name_translation.csv")

# Merge data
df = pd.merge(orders, items, on='order_id')
df = pd.merge(df, products, on='product_id')
df = pd.merge(df, categories, on='product_category_name', how='left')

# ================= STEP 5: Convert Dates =================
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'], errors='coerce')

# Create delivery time feature
df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

# ================= STEP 6: Clean Data =================
# Drop rows with missing essential columns
required = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
df = df.dropna(subset=required + ['delivery_time_days'])

# Drop unnecessary columns
drop_cols = [
    'order_id', 'customer_id', 'seller_id', 'product_id',
    'order_purchase_timestamp', 'order_delivered_customer_date',
    'order_estimated_delivery_date', 'order_approved_at',
    'shipping_limit_date', 'product_category_name', 'product_category_name_english'
]
df = df.drop(columns=drop_cols, errors='ignore')

# Drop any remaining rows with nulls
df = df.dropna()

# ================= STEP 7: Train Model =================
# Features and target
X = df.drop(columns=['delivery_time_days'])
y = df['delivery_time_days']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# ================= STEP 8: Evaluate Model =================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Delivery Time (Days)")
plt.ylabel("Predicted Delivery Time (Days)")
plt.title("Actual vs Predicted Delivery Time")
plt.grid(True)
plt.show()
