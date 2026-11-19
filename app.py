
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Hotel Cancellation Prediction",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("Enter the booking details to predict whether the booking will be cancelled.")

with st.form("booking_form"):

    st.subheader("Hotel and Arrival Details")

    col1, col2 = st.columns(2)

    with col1:
        hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
        arrival_date_year = st.number_input("Arrival Year", 2015, 2030, 2017)
        arrival_date_month = st.selectbox(
            "Arrival Month",
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]
        )
        arrival_date_week_number = st.number_input("Arrival Week Number", 1, 53, 27)
        arrival_date_day_of_month = st.number_input("Arrival Day", 1, 31, 15)
        country = st.text_input("Country", "PRT")

    with col2:
        lead_time = st.number_input("Lead Time", min_value=0, value=100)
        stays_in_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=2)
        stays_in_week_nights = st.number_input("Week Nights", min_value=0, value=3)
        adults = st.number_input("Adults", min_value=0, value=2)
        children = st.number_input("Children", min_value=0.0, value=0.0)
        babies = st.number_input("Babies", min_value=0, value=0)

    st.subheader("Booking Details")

    col3, col4 = st.columns(2)

    with col3:
        meal = st.selectbox("Meal", ["BB", "HB", "FB", "SC", "Undefined"])
        market_segment = st.selectbox(
            "Market Segment",
            ["Online TA", "Offline TA/TO", "Direct", "Corporate",
             "Groups", "Complementary", "Aviation", "Undefined"]
        )
        distribution_channel = st.selectbox(
            "Distribution Channel",
            ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
        )
        reserved_room_type = st.selectbox(
            "Reserved Room Type", list("ABCDEFGHIL")
        )
        assigned_room_type = st.selectbox(
            "Assigned Room Type", list("ABCDEFGHIL")
        )
        deposit_type = st.selectbox(
            "Deposit Type", ["No Deposit", "Non Refund", "Refundable"]
        )

    with col4:
        is_repeated_guest = st.selectbox("Repeated Guest?", [0, 1])
        previous_cancellations = st.number_input(
            "Previous Cancellations", min_value=0, value=0
        )
        previous_bookings_not_canceled = st.number_input(
            "Previous Bookings Not Cancelled", min_value=0, value=0
        )
        booking_changes = st.number_input("Booking Changes", min_value=0, value=0)
        agent = st.number_input("Agent ID", min_value=0.0, value=0.0)
        company = st.number_input("Company ID", min_value=0.0, value=0.0)

    st.subheader("Additional Details")

    col5, col6 = st.columns(2)

    with col5:
        days_in_waiting_list = st.number_input(
            "Days in Waiting List", min_value=0, value=0
        )
        customer_type = st.selectbox(
            "Customer Type",
            ["Transient", "Transient-Party", "Contract", "Group"]
        )
        adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, value=100.0)

    with col6:
        required_car_parking_spaces = st.number_input(
            "Required Car Parking Spaces", min_value=0, value=0
        )
        total_of_special_requests = st.number_input(
            "Total Special Requests", min_value=0, value=0
        )
        city = st.text_input("City", "Lisbon")

    submit = st.form_submit_button("Predict Cancellation")

if submit:

    booking_data = {
        "hotel": hotel,
        "lead_time": int(lead_time),
        "arrival_date_year": int(arrival_date_year),
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": int(arrival_date_week_number),
        "arrival_date_day_of_month": int(arrival_date_day_of_month),
        "stays_in_weekend_nights": int(stays_in_weekend_nights),
        "stays_in_week_nights": int(stays_in_week_nights),
        "adults": int(adults),
        "children": float(children),
        "babies": int(babies),
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": int(is_repeated_guest),
        "previous_cancellations": int(previous_cancellations),
        "previous_bookings_not_canceled": int(previous_bookings_not_canceled),
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "booking_changes": int(booking_changes),
        "deposit_type": deposit_type,
        "agent": float(agent),
        "company": float(company),
        "days_in_waiting_list": int(days_in_waiting_list),
        "customer_type": customer_type,
        "adr": float(adr),
        "required_car_parking_spaces": int(required_car_parking_spaces),
        "total_of_special_requests": int(total_of_special_requests),
        "city": city
    }

    try:
        with st.spinner("Predicting..."):
            response = requests.post(API_URL, json=booking_data, timeout=30)

        if response.status_code == 200:
            result = response.json()

            st.subheader("Prediction Result")

            if result["result"] == "Cancelled":
                st.error("⚠️ This booking is predicted to be CANCELLED.")
            else:
                st.success("✅ This booking is predicted to NOT BE CANCELLED.")

            st.write("Prediction:", result["prediction"])
            st.write("Result:", result["result"])

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI. Please start your backend.")

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
