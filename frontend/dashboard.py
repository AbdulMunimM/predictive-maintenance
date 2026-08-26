import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# API CONFIGURATION
# ============================================================

# Local backend for development/testing.
# For deployment, replace this with the deployed FastAPI URL
# through Streamlit secrets.
API_BASE_URL = "http://127.0.0.1:8000"

PREDICT_URL = f"{API_BASE_URL}/predict"
EXPLAIN_URL = f"{API_BASE_URL}/explain"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fa;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
        color: #1f2937;
    }

    .status-card {
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.4rem;
        border: 1px solid #e5e7eb;
        background: white;
    }

    .status-critical {
        border-left: 7px solid #dc2626;
    }

    .status-warning {
        border-left: 7px solid #f59e0b;
    }

    .status-normal {
        border-left: 7px solid #16a34a;
    }

    .status-title {
        font-size: 1.6rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .status-text {
        color: #4b5563;
        font-size: 1rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.2rem;
        min-height: 125px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 750;
    }

    .metric-subtext {
        color: #6b7280;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }

    .mode-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem;
        min-height: 125px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .mode-title {
        font-size: 1rem;
        font-weight: 700;
        color: #374151;
    }

    .mode-probability {
        font-size: 1.5rem;
        font-weight: 750;
        margin-top: 0.35rem;
        margin-bottom: 0.45rem;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 2rem;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 3rem;
        font-size: 1rem;
        font-weight: 650;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚙️ AI Predictive Maintenance</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine failure prediction, failure-mode classification, '
    'and explainable maintenance decisions'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MACHINE INPUTS
# ============================================================

st.markdown(
    '<div class="section-title">Machine Operating Conditions</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    machine_type = st.selectbox(
        "Machine Type",
        options=["L", "M", "H"],
        help="Machine type from the AI4I dataset."
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=0.0,
        value=300.0,
        step=0.1
    )

with col2:

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=0.0,
        value=310.0,
        step=0.1
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=0.0,
        value=1500.0,
        step=1.0
    )

with col3:

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0.0,
        value=100.0,
        step=1.0
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.write("")

analyze = st.button(
    "🔍 Analyze Machine",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION REQUEST
# ============================================================

if analyze:

    payload = {
        "Type": machine_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear
    }

    try:

        with st.spinner("Analyzing machine condition..."):

            response = requests.post(
                PREDICT_URL,
                json=payload,
                timeout=30
            )

            response.raise_for_status()

            st.session_state["result"] = response.json()
            st.session_state["last_payload"] = payload

            # Clear previous explanation when a new
            # machine is analyzed.
            st.session_state.pop(
                "explanation",
                None
            )

    except requests.exceptions.Timeout:

        st.error(
            "The predictive-maintenance API took too long to respond."
        )

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the FastAPI backend."
        )

        st.caption(
            f"Backend URL: {API_BASE_URL}"
        )

    except requests.exceptions.HTTPError as e:

        st.error(
            "The FastAPI backend returned an error."
        )

        st.caption(
            f"Technical details: {e}"
        )

    except requests.exceptions.RequestException as e:

        st.error(
            "An error occurred while contacting the API."
        )

        st.caption(
            f"Technical details: {e}"
        )


# ============================================================
# RESULTS
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()


    # ========================================================
    # MAINTENANCE STATUS
    # ========================================================

    status = result["maintenance_status"]

    if status == "CRITICAL":

        st.markdown(
            """
            <div class="status-card status-critical">
                <div class="status-title">
                    🔴 CRITICAL
                </div>
                <div class="status-text">
                    Machine failure risk detected.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif status == "FAILURE_DETECTED":

        st.markdown(
            """
            <div class="status-card status-warning">
                <div class="status-title">
                    🟠 FAILURE DETECTED
                </div>
                <div class="status-text">
                    The model predicts a machine failure,
                    but no known failure mode was identified.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="status-card status-normal">
                <div class="status-title">
                    🟢 NORMAL
                </div>
                <div class="status-text">
                    No machine failure is currently predicted.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # MAINTENANCE RECOMMENDATION
    # ========================================================

    st.markdown(
        '<div class="section-title">Maintenance Recommendation</div>',
        unsafe_allow_html=True
    )

    st.info(
        result["recommended_action"]
    )


    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">Prediction Summary</div>',
        unsafe_allow_html=True
    )

    summary1, summary2, summary3 = st.columns(3)


    # Failure Risk

    with summary1:

        failure_probability = (
            result["failure_probability"]
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Failure Risk
                </div>
                <div class="metric-value">
                    {failure_probability * 100:.2f}%
                </div>
                <div class="metric-subtext">
                    XGBoost failure classifier
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # Likely Failure Mode

    with summary2:

        predicted_modes = [
            mode
            for mode, data
            in result["failure_modes"].items()
            if data["predicted"]
        ]

        likely_mode = (
            ", ".join(predicted_modes)
            if predicted_modes
            else "None"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Likely Failure Mode
                </div>
                <div class="metric-value">
                    {likely_mode}
                </div>
                <div class="metric-subtext">
                    Multi-label failure classifier
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # Machine Status

    with summary3:

        machine_status = (
            "FAILURE RISK"
            if result["machine_failure"]
            else "NORMAL"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Machine Status
                </div>
                <div class="metric-value">
                    {machine_status}
                </div>
                <div class="metric-subtext">
                    Final maintenance decision
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # FAILURE RISK VISUALIZATION
    # ========================================================

    st.markdown(
        '<div class="section-title">Failure Risk Level</div>',
        unsafe_allow_html=True
    )

    st.progress(
        min(
            max(failure_probability, 0.0),
            1.0
        )
    )

    st.caption(
        "Failure probability estimated by Model 1."
    )


    # ========================================================
    # FAILURE MODE ANALYSIS
    # ========================================================

    st.markdown(
        '<div class="section-title">Failure Mode Analysis</div>',
        unsafe_allow_html=True
    )

    mode_columns = st.columns(4)

    for column, (mode, data) in zip(
        mode_columns,
        result["failure_modes"].items()
    ):

        with column:

            probability = data["probability"]

            st.markdown(
                f"""
                <div class="mode-card">
                    <div class="mode-title">
                        {mode}
                    </div>
                    <div class="mode-probability">
                        {probability * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(
                    max(probability, 0.0),
                    1.0
                )
            )

            if data["predicted"]:

                st.error("🔴 Predicted")

            else:

                st.success("🟢 Not predicted")


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">Model Explanation</div>',
        unsafe_allow_html=True
    )

    if result["machine_failure"]:

        explain_button = st.button(
            "🔎 Why was this machine flagged?",
            use_container_width=True
        )

        if explain_button:

            try:

                with st.spinner(
                    "Generating model explanation..."
                ):

                    explanation_response = requests.post(
                        EXPLAIN_URL,
                        json=st.session_state["last_payload"],
                        timeout=30
                    )

                    explanation_response.raise_for_status()

                    st.session_state["explanation"] = (
                        explanation_response.json()
                    )

            except requests.exceptions.Timeout:

                st.error(
                    "The explanation request timed out."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the explanation endpoint."
                )

            except requests.exceptions.RequestException as e:

                st.error(
                    "Unable to generate the model explanation."
                )

                st.caption(
                    f"Technical details: {e}"
                )


        # ----------------------------------------------------
        # Display explanation
        # ----------------------------------------------------

        if "explanation" in st.session_state:

            explanation = st.session_state["explanation"]

            st.info(
                "SHAP explains how the model's input features "
                "influenced this individual failure prediction. "
                "Positive values push the prediction toward "
                "failure; negative values push it away."
            )

            st.write(
                f"Base model score: "
                f"{explanation['base_value']:.3f}"
            )

            for item in explanation["contributions"]:

                feature_name = item["feature"]
                shap_value = item["shap_value"]

                # Make transformed feature names easier to read.
                feature_name = (
                    feature_name
                    .replace("num__", "")
                    .replace("cat__", "")
                )

                if shap_value >= 0:

                    st.error(
                        f"↑ {feature_name}: "
                        f"+{shap_value:.3f}"
                    )

                else:

                    st.success(
                        f"↓ {feature_name}: "
                        f"{shap_value:.3f}"
                    )

    else:

        st.info(
            "SHAP explanation is available when the machine "
            "is flagged for failure."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        AI4I 2020 Predictive Maintenance System
        · XGBoost · SHAP · FastAPI · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)