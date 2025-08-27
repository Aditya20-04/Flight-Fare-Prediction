import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Set page config with a wide layout and a custom title
st.set_page_config(page_title="Flight Fare Predictor", layout="wide")

# --- Custom CSS for the new design ---
st.markdown("""
<style>
    /* Main body and app container styling */
    .stApp {
        background: linear-gradient(135deg, #0a0e14, #141a23);
        color: #d1e2f4;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f77f00;
    }

    /* Style the main app title */
    .st-emotion-cache-18ni7ap {
        text-align: center;
        margin-top: -20px;
        background: -webkit-linear-gradient(45deg, #f77f00, #fcbf49);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: bold;
        padding-top: 20px;
    }

    /* Card styling for the input form */
    .st-emotion-cache-163v45g {
        background-color: #0d121c;
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #2e3b4a;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
    }
    
    /* Style for the sidebar radio buttons */
    .st-emotion-cache-1m6csg5 {
        color: #d1e2f4;
    }

    /* Style for the sidebar headers */
    .st-emotion-cache-1090x21 {
        color: #f77f00;
    }

    /* Styling for the new ticket display */
    .ticket-card {
        background-color: #0d121c;
        padding: 40px;
        border-radius: 20px;
        border: 1px solid #2e3b4a;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    
    .ticket-card h3 {
        color: #d1e2f4;
        margin-bottom: 20px;
        border-bottom: 2px dashed #2e3b4a;
        padding-bottom: 10px;
        width: 100%;
    }
    
    .ticket-details {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around;
        width: 100%;
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .detail-item {
        background-color: #1a232f;
        padding: 15px;
        border-radius: 10px;
        flex-grow: 1;
        min-width: 200px;
        text-align: left;
    }
    
    .detail-item b {
        color: #f77f00;
    }
    
    .price-display {
        background: linear-gradient(45deg, #fcbf49, #f77f00);
        padding: 20px 40px;
        border-radius: 15px;
        color: #0d121c;
        font-size: 2.5em;
        font-weight: bold;
        box-shadow: 0 6px 15px rgba(247, 127, 0, 0.4);
        margin-top: 20px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)


st.title("✈️ Flight Fare Prediction App")

# ---------------- Load dataset ---------------- #
@st.cache_data
def load_data():
    """Loads the dataset from a zip file and caches it."""
    with zipfile.ZipFile("Clean_Dataset.csv.zip", 'r') as z:
        csv_name = z.namelist()[0]
        df = pd.read_csv(z.open(csv_name))
    return df

df = load_data()
df = df.dropna()

# ---------------- Encode Categorical ---------------- #
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ---------------- Features & Target ---------------- #
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Model Selection ---------------- #
st.sidebar.header("⚙️ Choose Model")
model_choice = st.sidebar.radio("Select Model", ["Random Forest", "Decision Tree", "Linear Regression"])

# Initialize and fit the chosen model
if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Decision Tree":
    model = DecisionTreeRegressor(random_state=42)
else:
    model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------- Model Performance ---------------- #
st.sidebar.subheader("📊 Model Performance")
st.sidebar.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
st.sidebar.metric("RMSE", f"₹ {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# ---------------- Passenger Input Form ---------------- #
st.subheader("🧑‍✈️ Enter Passenger Flight Details")

with st.form("passenger_form"):
    # Split layout into two columns for a cleaner look
    col1, col2 = st.columns(2)
    
    with col1:
        # Airline
        airline = st.selectbox("Airline", label_encoders["airline"].classes_)
        airline_enc = label_encoders["airline"].transform([airline])[0]
    
        # Source City
        source = st.selectbox("Source City", label_encoders["source_city"].classes_)
        source_enc = label_encoders["source_city"].transform([source])[0]
    
        # Departure Time
        dep_cat = st.selectbox("Departure Time", label_encoders["departure_time"].classes_)
        dep_enc = label_encoders["departure_time"].transform([dep_cat])[0]
    
        # Total Stops
        total_stops = st.selectbox("Total Stops", sorted(df["stops"].unique()))
    
    with col2:
        # Date of Journey
        journey_date = st.date_input("Date of Journey", datetime.date.today())
        
        # Destination City
        destination = st.selectbox("Destination City", label_encoders["destination_city"].classes_)
        destination_enc = label_encoders["destination_city"].transform([destination])[0]
    
        # Arrival Time
        arr_cat = st.selectbox("Arrival Time", label_encoders["arrival_time"].classes_)
        arr_enc = label_encoders["arrival_time"].transform([arr_cat])[0]
    
        # Duration (hours input)
        duration = st.number_input("Duration (in hours)", min_value=1.0, max_value=30.0, step=0.5)

    # Travel class
    travel_class = st.radio("Travel Class", label_encoders["class"].classes_)
    class_enc = label_encoders["class"].transform([travel_class])[0]

    submitted = st.form_submit_button("🔮 Predict Fare")

# ---------------- Make Prediction ---------------- #
if submitted:
    today = datetime.date.today()
    days_left = (journey_date - today).days

    # Build input dict with SAME COLUMNS as training
    input_data = {
        "Unnamed: 0": 0,
        "airline": airline_enc,
        "flight": 0,  # dummy
        "source_city": source_enc,
        "destination_city": destination_enc,
        "class": class_enc,
        "departure_time": dep_enc,
        "arrival_time": arr_enc,
        "stops": total_stops,
        "duration": duration,
        "days_left": days_left
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[X.columns]

    prediction = model.predict(input_df)[0]

    # ---------------- Stylish Ticket Display (New Design) ---------------- #
    st.subheader("🎟️ Predicted Flight Fare")
    st.markdown(
        f"""
        <div class="ticket-card">
            <div class="ticket-details">
                <div class="detail-item"><b>Airline:</b> {airline}</div>
                <div class="detail-item"><b>Source:</b> {source}</div>
                <div class="detail-item"><b>Destination:</b> {destination}</div>
                <div class="detail-item"><b>Date:</b> {journey_date.strftime('%d %b %Y')}</div>
                <div class="detail-item"><b>Departure:</b> {dep_cat}</div>
                <div class="detail-item"><b>Arrival:</b> {arr_cat}</div>
                <div class="detail-item"><b>Class:</b> {travel_class}</div>
                <div class="detail-item"><b>Stops:</b> {total_stops}</div>
            </div>
            <div class="price-display">
                ₹ {round(prediction, 2)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
