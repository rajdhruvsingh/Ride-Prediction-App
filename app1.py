import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("🚖 Ride Prediction App")
st.divider()

@st.cache_data
def load_and_train_model():
    
    data = pd.read_csv("ncr_ride_bookings.csv")
    
    data = data.drop(columns=[
        'Date', 'Time', 'Booking ID', 'Customer ID',
        'Cancelled Rides by Customer', 'Reason for cancelling by Customer',
        'Cancelled Rides by Driver', 'Driver Cancellation Reason',
        'Incomplete Rides', 'Incomplete Rides Reason'
    ], errors='ignore')
    
    data = data.dropna(subset=['Booking Status'])
    
    X = data.drop(columns=['Booking Status'])
    y = data['Booking Status']
    
    numeric_cols = X.select_dtypes(include=['number']).columns
    
    X['data_missing_flag'] = X[numeric_cols].isna().any(axis=1).astype(int)
    
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    text_cols = X.select_dtypes(include=['object']).columns
    X[text_cols] = X[text_cols].fillna('Unknown')
    
    label_encoders = {}
    for col in text_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    y_test_encoded = le_target.transform(y_test)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train_encoded)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    return model, scaler, label_encoders, le_target, X.columns, accuracy

model, scaler, label_encoders, le_target, feature_cols, accuracy = load_and_train_model()


st.subheader("Enter Ride Details")

col1, col2 = st.columns(2)

with col1:
    vehicle_type = st.selectbox("Vehicle Type", label_encoders['Vehicle Type'].classes_)
    pickup_location = st.selectbox("Pickup Location", label_encoders['Pickup Location'].classes_)
    drop_location = st.selectbox("Drop Location", label_encoders['Drop Location'].classes_)
    avg_vtat = st.number_input("Avg VTAT (minutes)", min_value=0.0, value=5.0)
    avg_ctat = st.number_input("Avg CTAT (minutes)", min_value=0.0, value=5.0)

with col2:
    booking_value = st.number_input("Booking Value (₹)", min_value=0.0, value=100.0)
    ride_distance = st.number_input("Ride Distance (km)", min_value=0.0, value=5.0)
    driver_rating = st.number_input("Driver Rating", min_value=0.0, max_value=5.0, value=4.5)
    customer_rating = st.number_input("Customer Rating", min_value=0.0, max_value=5.0, value=4.5)
    payment_method = st.selectbox("Payment Method", label_encoders['Payment Method'].classes_)

user_data = pd.DataFrame({
    'Vehicle Type': [vehicle_type],
    'Pickup Location': [pickup_location],
    'Drop Location': [drop_location],
    'Avg VTAT': [avg_vtat],
    'Avg CTAT': [avg_ctat],
    'Booking Value': [booking_value],
    'Ride Distance': [ride_distance],
    'Driver Ratings': [driver_rating],
    'Customer Rating': [customer_rating],
    'Payment Method': [payment_method],
    'data_missing_flag': [0]
})

user_data['Vehicle Type'] = label_encoders['Vehicle Type'].transform(user_data['Vehicle Type'])
user_data['Pickup Location'] = label_encoders['Pickup Location'].transform(user_data['Pickup Location'])
user_data['Drop Location'] = label_encoders['Drop Location'].transform(user_data['Drop Location'])
user_data['Payment Method'] = label_encoders['Payment Method'].transform(user_data['Payment Method'])

if st.button("🔮 Predict Ride Outcome", use_container_width=True):
    user_data_scaled = scaler.transform(user_data)
    
    prediction = model.predict(user_data_scaled)[0]
    probability = model.predict_proba(user_data_scaled)[0]
    
    predicted_status = le_target.classes_[prediction]
    predicted_prob = probability[prediction]
    
    
    st.subheader("Prediction Result")
    
    if predicted_status == 'Completed':
        st.success(f"✅ **Ride might be Completed**")
    else:
        st.warning(f"⚠️ **Ride might be {predicted_status}**")
    

st.divider()

st.caption("🚖 Ride Prediction App | Created by Dhruv Raj Singh")