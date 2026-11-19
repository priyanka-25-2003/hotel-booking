from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Hotel Booking Prediction",
    page_icon="🏨",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "hotel_cancellation_model.pkl"


# =========================================================
# LOAD DATASET
# =========================================================
@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


try:
    df = load_dataset()
except FileNotFoundError:
    st.error(f"Dataset file not found: {DATASET_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()


# =========================================================
# COMMON STYLING
# =========================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #DCEBFF, #B8D8F8, #D6F5F5);
}
[data-testid="stMain"] {
    background: transparent;
}
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {
    color: #123B65;
    font-weight: 800;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1020 0%, #17143A 50%, #25134D 100%);
    border-right: 1px solid #39316D;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF;
}
[data-testid="stSidebar"] h1 {
    color: #A78BFA !important;
    font-weight: 800;
}
[data-testid="stSidebar"] .stRadio label {
    color: #E2E8F0 !important;
    font-size: 15px;
}
[data-testid="stSidebar"] input {
    background-color: #202044 !important;
    color: white !important;
    border: 1px solid #5145CD !important;
}
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(90deg, #5145CD, #9333EA);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: bold;
}
[data-testid="stMain"] .stButton > button {
    background: linear-gradient(135deg, #075985, #0891B2);
    color: white !important;
    border: 1px solid #67E8F9;
    border-radius: 12px;
    font-weight: bold;
}
[data-testid="stMain"] hr {
    border-color: #67B7D1;
}
.main-title {
    background: linear-gradient(90deg, #0B1020, #25134D, #5145CD);
    color: white;
    padding: 22px;
    border-radius: 12px;
    font-size: 27px;
    font-weight: bold;
    border-bottom: 3px solid #22D3EE;
}
.project-card {
    background: rgba(255,255,255,0.88);
    padding: 22px;
    border-radius: 15px;
    margin-bottom: 18px;
    border-left: 5px solid #C2185B;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
}
.project-card h3 {
    color: #880E4F !important;
}
.project-card p, .project-card li {
    color: #333333;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOGIN PAGE
# =========================================================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background:
        linear-gradient(rgba(0,0,0,0.50), rgba(0,0,0,0.50)),
        url("https://images.unsplash.com/photo-1566073771259-6a8506099945");
        background-size: cover;
        background-position: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏨 Hotel Booking System")
    st.header("🔐 Login Page")

    username = st.text_input("👤 Enter Username")
    password = st.text_input("🔑 Enter Password", type="password")

    if st.button("🚀 Login", width="stretch"):
        if username == "priyanka" and password == "priyanka":
            st.session_state.login = True
            st.rerun()
        else:
            st.error("❌ Invalid Username or Password")


# =========================================================
# MAIN DASHBOARD
# =========================================================
else:
    st.sidebar.title("🏨 HOTEL ADMIN")
    st.sidebar.text_input("🔍 Search", key="sidebar_search")
    st.sidebar.markdown("### MENU")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Analytics",
            "🔮 Prediction",
            "📋 Booking Details",
            "⚙️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.write("👤 Logged in as Priyanka")

    if st.sidebar.button("🚪 Logout", width="stretch"):
        st.session_state.login = False
        st.rerun()

    st.markdown(
        '<div class="main-title">🏨 HOTEL BOOKING MANAGEMENT SYSTEM</div>',
        unsafe_allow_html=True
    )
    st.markdown("🎉 Welcome to your Hotel Management Dashboard")
    st.markdown("---")

    # =====================================================
    # HOME PAGE
    # =====================================================
    if page == "🏠 Home":
        st.subheader("Dashboard Overview")

        total_bookings = len(df)
        cancelled = int(df["is_canceled"].sum()) if "is_canceled" in df.columns else 0
        confirmed = total_bookings - cancelled
        cancellation_rate = (cancelled / total_bookings * 100) if total_bookings else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏨 Total Bookings", f"{total_bookings:,}")
        col2.metric("❌ Cancelled", f"{cancelled:,}")
        col3.metric("✅ Confirmed", f"{confirmed:,}")
        col4.metric("📊 Cancellation Rate", f"{cancellation_rate:.2f}%")

        st.markdown("---")
        st.subheader("📌 Dataset Preview")
        st.dataframe(df.head(10), width="stretch", hide_index=True)

        if "is_canceled" in df.columns:
            status_df = pd.DataFrame({
                "Booking Status": ["Confirmed", "Cancelled"],
                "Bookings": [confirmed, cancelled]
            })
            fig = px.pie(
                status_df,
                names="Booking Status",
                values="Bookings",
                hole=0.5,
                title="Booking Status Overview",
                color="Booking Status",
                color_discrete_map={
                    "Confirmed": "#5145CD",
                    "Cancelled": "#F15B5B"
                }
            )
            st.plotly_chart(fig, width="stretch")

    # =====================================================
    # ANALYTICS PAGE
    # =====================================================
    elif page == "📊 Analytics":
        st.title("📊 Hotel Booking Analytics")
        st.caption("Booking and cancellation insights")

        data = df.copy()

        if "hotel" in data.columns:
            data["hotel_type"] = data["hotel"].astype(str).str.strip()

        st.subheader("🔍 Filter Your Data")
        filter_col1, filter_col2 = st.columns(2)

        selected_city = "All Cities"
        if "city" in data.columns:
            with filter_col1:
                selected_city = st.selectbox(
                    "Select City",
                    ["All Cities"] + sorted(data["city"].dropna().astype(str).unique().tolist())
                )
        else:
            with filter_col1:
                st.caption("City filter is unavailable because the dataset has no city column.")

        selected_hotel = "All Hotels"
        if "hotel_type" in data.columns:
            with filter_col2:
                selected_hotel = st.selectbox(
                    "Select Hotel",
                    ["All Hotels"] + sorted(data["hotel_type"].dropna().unique().tolist())
                )
        else:
            with filter_col2:
                st.caption("Hotel filter is unavailable.")

        filtered = data.copy()

        if selected_city != "All Cities" and "city" in filtered.columns:
            filtered = filtered[filtered["city"] == selected_city]

        if selected_hotel != "All Hotels" and "hotel_type" in filtered.columns:
            filtered = filtered[filtered["hotel_type"] == selected_hotel]

        if filtered.empty:
            st.warning("No booking records found for the selected filters.")
        else:
            total = len(filtered)
            cancelled = int(filtered["is_canceled"].sum()) if "is_canceled" in filtered.columns else 0
            confirmed = total - cancelled
            cancellation_rate = cancelled / total * 100 if total else 0

            st.markdown("---")
            st.subheader("📌 Booking Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Bookings", f"{total:,}")
            c2.metric("Cancelled", f"{cancelled:,}")
            c3.metric("Confirmed", f"{confirmed:,}")
            c4.metric("Cancellation Rate", f"{cancellation_rate:.2f}%")

            # Monthly booking trend
            if "arrival_date_month" in filtered.columns:
                month_order = [
                    "January", "February", "March", "April",
                    "May", "June", "July", "August",
                    "September", "October", "November", "December"
                ]
                monthly_source = filtered.copy()
                monthly_source["arrival_date_month"] = pd.Categorical(
                    monthly_source["arrival_date_month"].astype(str).str.strip(),
                    categories=month_order,
                    ordered=True
                )
                monthly = (
                    monthly_source.groupby("arrival_date_month", observed=False)
                    .size().reset_index(name="Bookings").dropna()
                )
                fig1 = px.bar(
                    monthly, x="arrival_date_month", y="Bookings",
                    color="Bookings", color_continuous_scale="Purples",
                    title="Month-wise Hotel Bookings"
                )
                st.plotly_chart(fig1, width="stretch")

            # Cancellation charts
            if "is_canceled" in filtered.columns:
                st.subheader("❌ Cancellation Analysis")
                c1, c2 = st.columns(2)

                with c1:
                    status_data = pd.DataFrame({
                        "Status": ["Confirmed", "Cancelled"],
                        "Count": [confirmed, cancelled]
                    })
                    fig2 = px.pie(
                        status_data, names="Status", values="Count",
                        hole=0.55, title="Booking Status Distribution",
                        color="Status",
                        color_discrete_map={
                            "Confirmed": "#5145CD",
                            "Cancelled": "#F15B5B"
                        }
                    )
                    st.plotly_chart(fig2, width="stretch")

                with c2:
                    if "arrival_date_month" in filtered.columns:
                        month_order = [
                            "January", "February", "March", "April",
                            "May", "June", "July", "August",
                            "September", "October", "November", "December"
                        ]
                        month_cancel_source = filtered.copy()
                        month_cancel_source["arrival_date_month"] = pd.Categorical(
                            month_cancel_source["arrival_date_month"].astype(str).str.strip(),
                            categories=month_order, ordered=True
                        )
                        monthly_cancel = (
                            month_cancel_source.groupby("arrival_date_month", observed=False)["is_canceled"]
                            .mean().reset_index().dropna()
                        )
                        monthly_cancel["Cancellation Rate"] = monthly_cancel["is_canceled"] * 100
                        fig3 = px.line(
                            monthly_cancel, x="arrival_date_month",
                            y="Cancellation Rate", markers=True,
                            title="Monthly Cancellation Percentage"
                        )
                        fig3.update_traces(line_color="#9333EA", line_width=3)
                        st.plotly_chart(fig3, width="stretch")

            # Hotel-wise analysis
            if "hotel" in filtered.columns and "is_canceled" in filtered.columns:
                st.subheader("🏨 Hotel-wise Booking Analysis")
                hotel_data = (
                    filtered.groupby("hotel")
                    .agg(
                        Total_Bookings=("is_canceled", "count"),
                        Cancelled=("is_canceled", "sum")
                    ).reset_index()
                )
                hotel_data["Confirmed"] = hotel_data["Total_Bookings"] - hotel_data["Cancelled"]
                fig4 = px.bar(
                    hotel_data, x="hotel",
                    y=["Confirmed", "Cancelled"], barmode="group",
                    title="Hotel-wise Confirmed and Cancelled Bookings",
                    color_discrete_sequence=["#5145CD", "#F15B5B"]
                )
                st.plotly_chart(fig4, width="stretch")

            # Customer type analysis
            if "customer_type" in filtered.columns:
                st.subheader("👥 Customer Analysis")
                c1, c2 = st.columns(2)

                with c1:
                    customer_data = (
                        filtered.groupby("customer_type")
                        .size().reset_index(name="Bookings")
                    )
                    fig5 = px.pie(
                        customer_data, names="customer_type",
                        values="Bookings", hole=0.5,
                        title="Customer Type Distribution"
                    )
                    st.plotly_chart(fig5, width="stretch")

                with c2:
                    if "market_segment" in filtered.columns:
                        market_data = (
                            filtered.groupby("market_segment")
                            .size().reset_index(name="Bookings")
                            .sort_values("Bookings", ascending=False)
                        )
                        fig6 = px.bar(
                            market_data, x="market_segment", y="Bookings",
                            color="Bookings", color_continuous_scale="Purples",
                            title="Market Segment Analysis"
                        )
                        st.plotly_chart(fig6, width="stretch")

            # Lead time analysis
            if "lead_time" in filtered.columns:
                st.subheader("⏳ Booking Lead Time Analysis")
                if "is_canceled" in filtered.columns:
                    fig7 = px.histogram(
                        filtered, x="lead_time", color="is_canceled",
                        nbins=30, barmode="overlay", opacity=0.7,
                        title="Lead Time vs Booking Cancellation",
                        color_discrete_map={0: "#5145CD", 1: "#F15B5B"}
                    )
                else:
                    fig7 = px.histogram(
                        filtered, x="lead_time", nbins=30,
                        title="Booking Lead Time Distribution"
                    )
                st.plotly_chart(fig7, width="stretch")

            # Deposit analysis
            if "deposit_type" in filtered.columns and "is_canceled" in filtered.columns:
                st.subheader("💳 Deposit Type Analysis")
                deposit_data = (
                    filtered.groupby("deposit_type")["is_canceled"]
                    .mean().reset_index()
                )
                deposit_data["Cancellation Rate"] = deposit_data["is_canceled"] * 100
                fig8 = px.bar(
                    deposit_data, x="deposit_type", y="Cancellation Rate",
                    color="Cancellation Rate", color_continuous_scale="Purples",
                    title="Cancellation Rate by Deposit Type"
                )
                st.plotly_chart(fig8, width="stretch")

            st.markdown("---")
            st.subheader("📋 Booking Dataset")
            st.dataframe(filtered, width="stretch", hide_index=True)
            st.download_button(
                "📥 Download Analytics Data",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="hotel_analytics.csv",
                mime="text/csv",
                width="stretch"
            )

    # =====================================================
    # AI PREDICTION PAGE
    # =====================================================
    elif page == "🔮 Prediction":
        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg,#FCE4EC,#F8BBD0,#F48FB1);
        }
        div[data-testid="stForm"] {
            background: rgba(255,255,255,0.85);
            padding: 20px;
            border-radius: 15px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.title("🔮 AI HOTEL BOOKING PREDICTION")
        st.write("Enter booking details to predict whether the booking will be cancelled.")

        with st.form("prediction_form"):
            st.subheader("🏨 Hotel and Arrival Details")
            col1, col2 = st.columns(2)

            with col1:
                hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
                lead_time = st.number_input("Lead Time", min_value=0, value=100)
                arrival_date_year = st.number_input("Arrival Year", 2015, 2030, 2017)
                arrival_date_month = st.selectbox(
                    "Arrival Month",
                    ["January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"]
                )
                arrival_date_week_number = st.number_input("Arrival Week Number", 1, 53, 27)
                arrival_date_day_of_month = st.number_input("Arrival Day", 1, 31, 15)
                country = st.text_input("Country Code", "PRT")

            with col2:
                stays_in_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=2)
                stays_in_week_nights = st.number_input("Week Nights", min_value=0, value=3)
                adults = st.number_input("Adults", min_value=0, value=2)
                children = st.number_input("Children", min_value=0.0, value=0.0)
                babies = st.number_input("Babies", min_value=0, value=0)
                meal = st.selectbox("Meal", ["BB", "HB", "FB", "SC", "Undefined"])

            st.subheader("📋 Booking Details")
            col3, col4 = st.columns(2)

            with col3:
                market_segment = st.selectbox(
                    "Market Segment",
                    ["Online TA", "Offline TA/TO", "Direct", "Corporate",
                     "Groups", "Complementary", "Aviation", "Undefined"]
                )
                distribution_channel = st.selectbox(
                    "Distribution Channel",
                    ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
                )
                is_repeated_guest = st.selectbox("Repeated Guest", [0, 1])
                previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
                previous_bookings_not_canceled = st.number_input(
                    "Previous Bookings Not Cancelled", min_value=0, value=0
                )

            with col4:
                reserved_room_type = st.selectbox(
                    "Reserved Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
                )
                assigned_room_type = st.selectbox(
                    "Assigned Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
                )
                booking_changes = st.number_input("Booking Changes", min_value=0, value=0)
                deposit_type = st.selectbox(
                    "Deposit Type", ["No Deposit", "Non Refund", "Refundable"]
                )
                agent = st.number_input("Agent ID", min_value=0.0, value=0.0)

            st.subheader("👥 Customer and Additional Details")
            col5, col6 = st.columns(2)

            with col5:
                days_in_waiting_list = st.number_input("Waiting List Days", min_value=0, value=0)
                customer_type = st.selectbox(
                    "Customer Type", ["Transient", "Transient-Party", "Contract", "Group"]
                )
                adr = st.number_input("Average Daily Rate", min_value=0.0, value=100.0)

            with col6:
                required_car_parking_spaces = st.number_input("Parking Spaces", min_value=0, value=0)
                total_of_special_requests = st.number_input("Special Requests", min_value=0, value=1)

            submit = st.form_submit_button(
                "🔮 PREDICT CANCELLATION",
                width="stretch"
            )

        # IMPORTANT: This is outside the form, but inside the Prediction page.
        if submit:
            try:
                if not MODEL_PATH.exists():
                    st.error(
                        "Model file not found. Run basic_model_fast.py first "
                        "and make sure hotel_cancellation_model.pkl is in the project folder."
                    )
                else:
                    model = joblib.load(MODEL_PATH)

                    booking_data = {
                        "hotel": hotel,
                        "lead_time": lead_time,
                        "arrival_date_year": arrival_date_year,
                        "arrival_date_month": arrival_date_month,
                        "arrival_date_week_number": arrival_date_week_number,
                        "arrival_date_day_of_month": arrival_date_day_of_month,
                        "country": country.strip().upper(),
                        "stays_in_weekend_nights": stays_in_weekend_nights,
                        "stays_in_week_nights": stays_in_week_nights,
                        "adults": adults,
                        "children": children,
                        "babies": babies,
                        "meal": meal,
                        "market_segment": market_segment,
                        "distribution_channel": distribution_channel,
                        "is_repeated_guest": is_repeated_guest,
                        "previous_cancellations": previous_cancellations,
                        "previous_bookings_not_canceled": previous_bookings_not_canceled,
                        "reserved_room_type": reserved_room_type,
                        "assigned_room_type": assigned_room_type,
                        "booking_changes": booking_changes,
                        "deposit_type": deposit_type,
                        "agent": agent,
                        "days_in_waiting_list": days_in_waiting_list,
                        "customer_type": customer_type,
                        "adr": adr,
                        "required_car_parking_spaces": required_car_parking_spaces,
                        "total_of_special_requests": total_of_special_requests
                    }

                    input_data = pd.DataFrame([booking_data])

                    # Align user input with the model's original training columns.
                    expected_columns = getattr(model, "feature_names_in_", None)
                    if expected_columns is not None:
                        input_data = input_data.reindex(columns=list(expected_columns))

                    if "arrival_date_month" in input_data.columns:
                        input_data["arrival_date_month"] = (
                            input_data["arrival_date_month"].astype("string").str.strip()
                        )

                    prediction = int(model.predict(input_data)[0])

                    st.markdown("---")
                    st.subheader("📊 HOTEL BOOKING PREDICTION RESULT")

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_data)[0]
                        classes = list(model.classes_)
                        cancel_probability = float(
                            probabilities[classes.index(1)]
                        ) if 1 in classes else 0.0

                        result_col1, result_col2 = st.columns(2)
                        result_col1.metric(
                            "🤖 Prediction",
                            "Cancelled" if prediction == 1 else "Confirmed"
                        )
                        result_col2.metric(
                            "Cancellation Probability",
                            f"{cancel_probability * 100:.2f}%"
                        )
                        st.progress(max(0.0, min(1.0, cancel_probability)))
                    else:
                        st.metric(
                            "🤖 Prediction",
                            "Cancelled" if prediction == 1 else "Confirmed"
                        )

                    if prediction == 1:
                        st.error("❌ Booking is predicted to be CANCELLED")
                    else:
                        st.success("✅ Booking is predicted to be CONFIRMED")

            except Exception as e:
                st.error(f"Prediction Error: {e}")

    # =====================================================
    # BOOKING DETAILS PAGE
    # =====================================================
    elif page == "📋 Booking Details":
        st.title("📋 Hotel Booking Details")
        st.markdown("View, search and filter all hotel booking records.")

        booking_df = df.copy()

        st.metric("Total Bookings", f"{len(booking_df):,}")
        st.divider()

        search = st.text_input("🔍 Search Booking Details")

        col1, col2 = st.columns(2)

        with col1:
            if "hotel" in booking_df.columns:
                hotel_filter = st.selectbox(
                    "Select Hotel",
                    ["All"] + sorted(booking_df["hotel"].dropna().astype(str).unique().tolist())
                )
            else:
                hotel_filter = "All"
                st.caption("Hotel filter is unavailable.")

        with col2:
            if "is_canceled" in booking_df.columns:
                status_filter = st.selectbox(
                    "Booking Status", ["All", "Confirmed", "Cancelled"]
                )
            else:
                status_filter = "All"
                st.caption("Booking status filter is unavailable.")

        filtered_df = booking_df.copy()

        if hotel_filter != "All" and "hotel" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["hotel"] == hotel_filter]

        if status_filter == "Confirmed" and "is_canceled" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["is_canceled"] == 0]
        elif status_filter == "Cancelled" and "is_canceled" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["is_canceled"] == 1]

        if search:
            search_mask = filtered_df.astype(str).apply(
                lambda column: column.str.contains(search, case=False, na=False)
            ).any(axis=1)
            filtered_df = filtered_df[search_mask]

        total_filtered = len(filtered_df)
        st.subheader("📌 Booking Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Records Found", f"{total_filtered:,}")

        if "is_canceled" in filtered_df.columns:
            c2.metric(
                "Confirmed Bookings",
                f"{int((filtered_df['is_canceled'] == 0).sum()):,}"
            )
            c3.metric(
                "Cancelled Bookings",
                f"{int((filtered_df['is_canceled'] == 1).sum()):,}"
            )
        else:
            c2.metric("Confirmed Bookings", "N/A")
            c3.metric("Cancelled Bookings", "N/A")

        st.divider()
        st.subheader("🏨 Booking Records")
        st.dataframe(filtered_df, width="stretch", height=450, hide_index=True)

        st.download_button(
            "📥 Download Booking Details",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="hotel_booking_details.csv",
            mime="text/csv",
            width="stretch"
        )

    # =====================================================
    # ABOUT PROJECT PAGE
    # =====================================================
    elif page == "⚙️ About Project":
        st.title("🏨 Hotel Booking Cancellation Prediction")
        st.markdown("### ⚙️ Project Details & Overview")

        st.markdown("""
        <div class="project-card">
            <h3>📌 Project Overview</h3>
            <p>
            The Hotel Booking Cancellation Prediction System is a
            Machine Learning based application designed to predict
            whether a hotel booking will be cancelled or confirmed.
            It analyzes booking information and helps hotels understand
            cancellation patterns and improve reservation management.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="project-card">
            <h3>🎯 Project Objectives</h3>
            <ul>
                <li>Predict hotel booking cancellations.</li>
                <li>Analyze hotel booking patterns.</li>
                <li>Identify important cancellation factors.</li>
                <li>Help hotels improve reservation planning.</li>
                <li>Provide interactive visualizations and predictions.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("💻 Technologies Used")
        col1, col2, col3 = st.columns(3)
        col1.metric("Programming", "Python")
        col2.metric("Dashboard", "Streamlit")
        col3.metric("Visualization", "Plotly")

        col1, col2, col3 = st.columns(3)
        col1.metric("Data Processing", "Pandas")
        col2.metric("Machine Learning", "Scikit-learn")
        col3.metric("Model Saving", "Joblib")

        st.divider()
        st.subheader("🤖 Machine Learning Algorithms")
        for algorithm in [
            "Random Forest Classifier",
            "Decision Tree Classifier",
            "AdaBoost Classifier"
        ]:
            st.markdown(f"✅ **{algorithm}**")

        st.divider()
        st.subheader("🚀 Project Features")
        col1, col2 = st.columns(2)

        with col1:
            st.info("🏠 **Home Dashboard**\n\nBooking overview and summary.")
            st.info("📊 **Analytics**\n\nInteractive charts and booking analysis.")
            st.info("🔮 **AI Prediction**\n\nPredict cancellation using booking details.")

        with col2:
            st.info("📋 **Booking Details**\n\nSearch, filter and view booking records.")
            st.info("📥 **Download Data**\n\nDownload filtered booking records.")
            st.info("⚙️ **Project Details**\n\nProject information and technologies.")

        st.divider()
        st.subheader("🔄 Project Workflow")
        st.markdown("""
        1. 📂 Collect hotel booking dataset.
        2. 🧹 Clean and preprocess the data.
        3. 📊 Perform Exploratory Data Analysis.
        4. 🤖 Train Machine Learning models.
        5. 🏆 Select the best-performing model.
        6. 🔮 Predict booking cancellation.
        7. 📈 Display results in Streamlit dashboard.
        """)

        st.divider()
        st.success("🎉 Hotel Booking Cancellation Prediction System")
        st.caption("Machine Learning | Data Analysis | Interactive Dashboard")
