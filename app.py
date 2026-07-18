import streamlit as st
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Suicide Risk Assessment Agent",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.stApp{
    background-color:black;
}

h1,h2,h3{
    color:#003366;
}

.result{
    padding:15px;
    border-radius:10px;
    font-size:20px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("🧠 Suicide Detection AI Agent")

st.write("Enter your thoughts below to analyze emotional risk.")

# -----------------------------
# User Input
# -----------------------------
user_text = st.text_area(
    "Type your message",
    height=180
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Analyze"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")

    else:

        text_vector = vectorizer.transform([user_text])

        prediction = model.predict(text_vector)

        if prediction[0] == "suicide":

            st.error("⚠ HIGH RISK DETECTED")

            st.markdown("""
### Please Remember ❤️

You may be going through a difficult time.

Consider talking to:

- 👨‍👩‍👧 Family
- 🧑 Friends
- 🩺 Mental Health Professional

You don't have to face everything alone.
""")

            st.markdown("---")

            st.subheader("🧘 Stress Relief Exercises")

            st.write("""
### 1. Deep Breathing
- Inhale for 4 seconds
- Hold for 4 seconds
- Exhale for 6 seconds
- Repeat 5 times

### 2. Meditation
Sit quietly for 10 minutes and focus on your breathing.

### 3. Yoga
- ✔ Child's Pose
- ✔ Cat-Cow Stretch
- ✔ Cobra Pose
- ✔ Mountain Pose

Practice for 10–15 minutes.

### 4. Walk
Go outside for 15–20 minutes.

### 5. Drink Water
Stay hydrated.

### 6. Sleep
Aim for 7–8 hours of sleep.
""")

        else:

            st.success("✅ LOW RISK")

            st.markdown("""
Keep maintaining a healthy lifestyle.

### Daily Habits

✔ Exercise

✔ Meditation

✔ Healthy Food

✔ Proper Sleep

✔ Talk with Loved Ones

✔ Stay Positive ❤️
""")
