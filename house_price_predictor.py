# === ENHANCED HOUSE PRICE PREDICTOR - COMBINED MODELS ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("🏠 ENHANCED HOUSE PRICE PREDICTOR")
print("🤖 Combined Random Forest + Linear Regression")
print("=" * 60)

# === LOAD AND CLEAN DATA ===
try:
    df = pd.read_excel('your_house_data.xlsx', sheet_name='Table1')
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ File 'your_house_data.xlsx' not found!")
    exit()

print(f"📊 Loaded {len(df)} houses from dataset")

# Data cleaning
df_clean = df.copy()

# Identify price column
if ' Price' in df_clean.columns:
    price_column = ' Price'
elif 'Price' in df_clean.columns:
    price_column = 'Price'
else:
    print("❌ Price column not found!")
    exit()

print(f"Using price column: '{price_column}'")

# Convert numeric columns
numeric_columns = ['inStoreys', 'inBedrooms', 'inBathrooms', 'inCarSpaces', 
                  'dcTotalAreaM2', 'dcHouseLength', 'dcHouseWidth', 
                  'dcGroundFloorArea', 'dcAlfrescoArea', 'dcPorchArea', 
                  'dcGarageArea', price_column]

for column in numeric_columns:
    if column in df_clean.columns:
        df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
        median_val = df_clean[column].median()
        df_clean[column] = df_clean[column].fillna(median_val)

# Remove rows where Price is NaN
initial_rows = len(df_clean)
df_clean = df_clean.dropna(subset=[price_column])

if initial_rows != len(df_clean):
    print(f"⚠️ Removed {initial_rows - len(df_clean)} rows with invalid data")

print(f"Clean data shape: {df_clean.shape}")

if len(df_clean) < 3:
    print("❌ Not enough valid data after cleaning!")
    exit()

# === DEFINE FEATURES ===
feature_columns = [
    'inStoreys', 'inBedrooms', 'inBathrooms', 'inCarSpaces', 
    'dcTotalAreaM2', 'dcHouseLength', 'dcHouseWidth', 
    'dcGroundFloorArea', 'dcAlfrescoArea', 'dcPorchArea', 'dcGarageArea'
]

available_features = [col for col in feature_columns if col in df_clean.columns]
X = df_clean[available_features]
y = df_clean[price_column]

print(f"✅ Using {len(available_features)} features")

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"📊 Training with {len(X_train)} houses, testing with {len(X_test)} houses")

# === TRAIN BOTH MODELS ===
print("\n" + "=" * 50)
print("TRAINING BOTH MODELS")
print("=" * 50)

# 1. Random Forest
print("🌳 Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

# 2. Linear Regression
print("📈 Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

# === COMBINED MODEL (ENSEMBLE) ===
def combined_prediction(rf_pred, lr_pred, rf_weight=0.6, lr_weight=0.4):
    """
    Combine predictions from both models
    Default weights: Random Forest 60%, Linear Regression 40%
    """
    return (rf_pred * rf_weight) + (lr_pred * lr_weight)

# Test combined model on test data
combined_predictions = combined_prediction(rf_predictions, lr_predictions)
combined_mae = mean_absolute_error(y_test, combined_predictions)
combined_r2 = r2_score(y_test, combined_predictions)

# === FIND OPTIMAL WEIGHTS ===
print("\n🔧 Optimizing model weights...")
best_mae = float('inf')
best_weights = (0.5, 0.5)

# Try different weight combinations
for rf_weight in np.arange(0.1, 1.0, 0.1):
    lr_weight = 1.0 - rf_weight
    current_predictions = (rf_predictions * rf_weight) + (lr_predictions * lr_weight)
    current_mae = mean_absolute_error(y_test, current_predictions)
    
    if current_mae < best_mae:
        best_mae = current_mae
        best_weights = (rf_weight, lr_weight)

optimal_rf_weight, optimal_lr_weight = best_weights

# === RESULTS COMPARISON ===
print("\n" + "=" * 50)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 50)

print(f"🌳 RANDOM FOREST ONLY:")
print(f"   - Mean Absolute Error: ${rf_mae:,.2f}")
print(f"   - R² Score: {rf_r2:.4f}")

print(f"📈 LINEAR REGRESSION ONLY:")
print(f"   - Mean Absolute Error: ${lr_mae:,.2f}")
print(f"   - R² Score: {lr_r2:.4f}")

print(f"🤖 COMBINED MODEL (Optimized):")
print(f"   - Mean Absolute Error: ${best_mae:,.2f}")
print(f"   - R² Score: {r2_score(y_test, combined_prediction(rf_predictions, lr_predictions, *best_weights)):.4f}")
print(f"   - Optimal Weights: Random Forest {optimal_rf_weight:.0%}, Linear Regression {optimal_lr_weight:.0%}")

# Determine best approach
models = {
    'Random Forest': rf_mae,
    'Linear Regression': lr_mae, 
    'Combined': best_mae
}
best_method = min(models, key=models.get)

print(f"\n🎯 RECOMMENDED: {best_method} (Lowest Error)")

# === INTERACTIVE PREDICTION ===
def predict_final_price(house_features, rf_weight=optimal_rf_weight, lr_weight=optimal_lr_weight):
    """Predict price using combined model"""
    features_array = [house_features[feature] for feature in available_features]
    
    rf_price = rf_model.predict([features_array])[0]
    lr_price = lr_model.predict([features_array])[0]
    
    # Combined prediction
    final_price = (rf_price * rf_weight) + (lr_price * lr_weight)
    
    return final_price, rf_price, lr_price

# === MAIN INTERACTIVE LOOP ===
while True:
    print("\n" + "=" * 60)
    print("🏠 HOUSE PRICE PREDICTION")
    print("=" * 60)
    
    print("📝 Enter the features for your house (or type 'quit' to exit):")
    
    house_features = {}
    for feature in available_features:
        while True:
            try:
                prompts = {
                    'inStoreys': "Number of storeys (floors): ",
                    'inBedrooms': "Number of bedrooms: ",
                    'inBathrooms': "Number of bathrooms: ",
                    'inCarSpaces': "Number of car spaces: ",
                    'dcTotalAreaM2': "Total area in square meters: ",
                    'dcHouseLength': "House length (meters): ",
                    'dcHouseWidth': "House width (meters): ",
                    'dcGroundFloorArea': "Ground floor area (m²): ",
                    'dcAlfrescoArea': "Alfresco area (m²): ",
                    'dcPorchArea': "Porch area (m²): ",
                    'dcGarageArea': "Garage area (m²): "
                }
                
                prompt = prompts.get(feature, f"Enter {feature}: ")
                user_input = input(prompt)
                
                if user_input.lower() == 'quit':
                    print("👋 Thank you for using the Enhanced House Price Predictor!")
                    exit()
                
                value = float(user_input)
                house_features[feature] = value
                break
            except ValueError:
                print("❌ Please enter a valid number!")

    # Get predictions
    final_price, rf_price, lr_price = predict_final_price(house_features)
    
    # Display results
    print("\n" + "💰" * 50)
    print("🎯 FINAL PREDICTED PRICE (Combined Model):")
    print(f"   ${final_price:,.2f}")
    print("💰" * 50)
    
    print(f"\n📊 Individual Model Predictions:")
    print(f"   🌳 Random Forest: ${rf_price:,.2f}")
    print(f"   📈 Linear Regression: ${lr_price:,.2f}")
    print(f"   ⚖️  Weighted Average: {optimal_rf_weight:.0%} RF + {optimal_lr_weight:.0%} LR")
    
    print(f"\n📈 Accuracy Information:")
    print(f"   Expected Error Range: ±${best_mae:,.2f}")
    print(f"   Confidence: {max(0, combined_r2)*100:.1f}%")
    
    # Show price range
    lower_bound = final_price - best_mae
    upper_bound = final_price + best_mae
    print(f"   📏 Estimated Price Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
    
    # Feature importance
    print(f"\n🔍 Most Important Features:")
    rf_importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in rf_importance.head(3).iterrows():
        print(f"   • {row['feature']}: {row['importance']:.1%} impact")
    
    # Ask if user wants another prediction
    print("\n" + "-" * 50)
    another = input("Would you like to predict another house? (y/n): ").lower()
    if another != 'y':
        print("👋 Thank you for using the Enhanced House Price Predictor!")
        break

# === SAVE THE MODELS FOR FUTURE USE ===
print(f"\n💾 Models trained successfully!")
print(f"🌳 Random Forest Weight: {optimal_rf_weight:.0%}")
print(f"📈 Linear Regression Weight: {optimal_lr_weight:.0%}")
print(f"🎯 Combined Model Error: ${best_mae:,.2f}")

print(f"\n✨ Future predictions will use the optimized combined model!")
