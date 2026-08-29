from pathlib import Path
from datetime import datetime

import pandas as pd
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
# CONFIGURATION
# ============================================================

API_BASE_URL = st.secrets["API_BASE_URL"]

PREDICT_URL = f"{API_BASE_URL}/predict"
EXPLAIN_URL = f"{API_BASE_URL}/explain"

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "ai4i2020.csv"


# ============================================================
# LOAD SENSOR DATA
# ============================================================

@st.cache_data
def load_sensor_data():

    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required dataset columns: {missing_columns}"
        )

    return data


try:
    sensor_data = load_sensor_data()

except Exception as e:
    st.error("Unable to load the AI4I sensor dataset.")
    st.caption(f"Technical details: {e}")
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "mode": "Manual Analysis",
    "result": None,
    "last_payload": None,
    "explanation": None,

    # Manual history
    "manual_history": [],

    # Sensor replay
    "monitoring": False,
    "current_row": 0,
    "sensor_history": [],
    "replay_speed": 2
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


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
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.35rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1f2937;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }

    .status-card {
        background: white;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0 1.2rem 0;
        border: 1px solid #e5e7eb;
    }

    .status-critical {
        border-left: 8px solid #dc2626;
    }

    .status-warning {
        border-left: 8px solid #f59e0b;
    }

    .status-normal {
        border-left: 8px solid #16a34a;
    }

    .status-title {
        font-size: 1.6rem;
        font-weight: 750;
    }

    .status-text {
        color: #6b7280;
        margin-top: 0.3rem;
    }

    .sensor-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem;
        min-height: 100px;
        margin-bottom: 0.7rem;
    }

    .sensor-label {
        color: #6b7280;
        font-size: 0.84rem;
    }

    .sensor-value {
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.1rem;
        min-height: 115px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.85rem;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 750;
        margin-top: 0.3rem;
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
# OPERATION MODE
# ============================================================

st.markdown(
    '<div class="section-title">Operation Mode</div>',
    unsafe_allow_html=True
)

selected_mode = st.radio(
    "Choose how machine data will be provided",
    ["Manual Analysis", "Sensor Stream Replay"],
    horizontal=True,
    key="mode_selector"
)


if selected_mode != st.session_state.mode:

    st.session_state.mode = selected_mode
    st.session_state.result = None
    st.session_state.last_payload = None
    st.session_state.explanation = None

    st.session_state.monitoring = False

    if selected_mode == "Sensor Stream Replay":

        st.session_state.current_row = 0
        st.session_state.sensor_history = []


# ============================================================
# SHARED PREDICTION DISPLAY
# ============================================================

def display_prediction(result):

    # --------------------------------------------------------
    # Maintenance status
    # --------------------------------------------------------

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
                    The model predicts a machine failure.
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


    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Maintenance Recommendation'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        result["recommended_action"]
    )


    # --------------------------------------------------------
    # Prediction summary
    # --------------------------------------------------------

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
        '<div class="section-title">'
        'Prediction Summary'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Failure Risk
                </div>
                <div class="metric-value">
                    {result["failure_probability"] * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Likely Failure Mode
                </div>
                <div class="metric-value">
                    {likely_mode}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

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
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # Failure risk
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Failure Risk'
        '</div>',
        unsafe_allow_html=True
    )

    st.progress(
        min(
            max(result["failure_probability"], 0.0),
            1.0
        )
    )


    # --------------------------------------------------------
    # Failure modes
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Failure Mode Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    mode_columns = st.columns(4)

    for column, (mode, data) in zip(
        mode_columns,
        result["failure_modes"].items()
    ):

        with column:

            probability = data["probability"]

            st.metric(
                mode,
                f"{probability * 100:.2f}%"
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


# ============================================================
# MANUAL ANALYSIS MODE
# ============================================================

if selected_mode == "Manual Analysis":

    st.markdown(
        '<div class="section-title">'
        'Manual Machine Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Enter a specific machine reading for one-time analysis."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"],
            key="manual_type"
        )

        air_temperature = st.number_input(
            "Air Temperature [K]",
            min_value=0.0,
            value=300.0,
            step=0.1,
            key="manual_air"
        )

    with col2:

        process_temperature = st.number_input(
            "Process Temperature [K]",
            min_value=0.0,
            value=310.0,
            step=0.1,
            key="manual_process"
        )

        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=0.0,
            value=1500.0,
            step=1.0,
            key="manual_speed"
        )

    with col3:

        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            value=40.0,
            step=0.1,
            key="manual_torque"
        )

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0.0,
            value=100.0,
            step=1.0,
            key="manual_wear"
        )

    st.write("")

    analyze = st.button(
        "🔍 Analyze Machine",
        type="primary",
        use_container_width=True,
        key="manual_analyze"
    )

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

            with st.spinner(
                "Analyzing machine condition..."
            ):

                response = requests.post(
                    PREDICT_URL,
                    json=payload,
                    timeout=60
                )

                response.raise_for_status()

                result = response.json()


            st.session_state.result = result
            st.session_state.last_payload = payload
            st.session_state.explanation = None


            # ------------------------------------------------
            # Manual history
            # ------------------------------------------------

            predicted_modes = [
                mode
                for mode, data
                in result["failure_modes"].items()
                if data["predicted"]
            ]

            manual_record = {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Machine Type": machine_type,
                "Air Temp [K]": air_temperature,
                "Process Temp [K]": process_temperature,
                "Speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool Wear [min]": tool_wear,
                "Failure Risk": (
                    result["failure_probability"] * 100
                ),
                "Failure": (
                    "YES"
                    if result["machine_failure"]
                    else "NO"
                ),
                "Failure Mode": (
                    ", ".join(predicted_modes)
                    if predicted_modes
                    else "-"
                ),
                "Status": result["maintenance_status"]
            }

            st.session_state.manual_history.append(
                manual_record
            )


        except requests.exceptions.Timeout:

            st.error(
                "The predictive-maintenance API took too long to respond."
            )


        except requests.exceptions.RequestException as e:

            st.error(
                "Unable to connect to the predictive-maintenance API."
            )

            st.caption(
                f"Technical details: {e}"
            )


    # ========================================================
    # CURRENT MANUAL RESULT
    # ========================================================

    if st.session_state.result is not None:

        result = st.session_state.result

        display_prediction(result)


        # ====================================================
        # MANUAL HISTORY
        # ====================================================

        if st.session_state.manual_history:

            st.divider()

            st.markdown(
                '<div class="section-title">'
                'Manual Analysis History'
                '</div>',
                unsafe_allow_html=True
            )

            st.caption(
                f"{len(st.session_state.manual_history)} "
                "manual analyses performed in this session."
            )

            history_df = pd.DataFrame(
                st.session_state.manual_history
            )

            history_df["Failure Risk"] = (
                history_df["Failure Risk"]
                .map(lambda x: f"{x:.2f}%")
            )

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        if result["machine_failure"]:

            st.divider()

            st.markdown(
                '<div class="section-title">'
                'Model Explanation'
                '</div>',
                unsafe_allow_html=True
            )

            explain_button = st.button(
                "🔎 Why was this machine flagged?",
                use_container_width=True,
                key="manual_explain"
            )

            if explain_button:

                try:

                    with st.spinner(
                        "Generating model explanation..."
                    ):

                        response = requests.post(
                            EXPLAIN_URL,
                            json=st.session_state.last_payload,
                            timeout=60
                        )

                        response.raise_for_status()

                        st.session_state.explanation = (
                            response.json()
                        )

                except requests.exceptions.RequestException as e:

                    st.error(
                        "Unable to generate the model explanation."
                    )

                    st.caption(
                        f"Technical details: {e}"
                    )


            if st.session_state.explanation is not None:

                explanation = st.session_state.explanation

                st.info(
                    "SHAP explains how individual features "
                    "influenced the machine-failure prediction."
                )

                st.write(
                    f"Base model score: "
                    f"{explanation['base_value']:.3f}"
                )

                for item in explanation["contributions"]:

                    feature_name = (
                        item["feature"]
                        .replace("num__", "")
                        .replace("cat__", "")
                    )

                    shap_value = item["shap_value"]

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


# ============================================================
# SENSOR STREAM REPLAY MODE
# ============================================================

else:

    st.markdown(
        '<div class="section-title">'
        'Automatic Sensor Stream'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Historical AI4I records are replayed as simulated "
        "machine sensor readings."
    )

    control1, control2, control3, control4 = st.columns(4)


    # --------------------------------------------------------
    # Start / Stop
    # --------------------------------------------------------

    with control1:

        if st.session_state.monitoring:

            if st.button(
                "⏹ Stop Monitoring",
                use_container_width=True,
                key="stop_monitoring"
            ):

                st.session_state.monitoring = False
                st.rerun()

        else:

            if st.button(
                "▶ Start Monitoring",
                type="primary",
                use_container_width=True,
                key="start_monitoring"
            ):

                st.session_state.monitoring = True
                st.session_state.current_row = 0
                st.session_state.sensor_history = []
                st.session_state.result = None
                st.session_state.explanation = None
                st.rerun()


    # --------------------------------------------------------
    # Restart
    # --------------------------------------------------------

    with control2:

        if st.button(
            "↺ Restart Stream",
            use_container_width=True,
            key="restart_stream"
        ):

            st.session_state.monitoring = False
            st.session_state.current_row = 0
            st.session_state.sensor_history = []
            st.session_state.result = None
            st.session_state.explanation = None

            st.rerun()


    # --------------------------------------------------------
    # Replay speed
    # --------------------------------------------------------

    with control3:

        st.session_state.replay_speed = st.selectbox(
            "Replay Speed",
            [1, 2, 3, 5],
            index=1,
            format_func=lambda x: f"Every {x} sec",
            key="speed_selector"
        )


    # --------------------------------------------------------
    # Records
    # --------------------------------------------------------

    with control4:

        st.metric(
            "Sensor Records",
            f"{len(sensor_data):,}"
        )


# ============================================================
# SENSOR REPLAY FRAGMENT
# ============================================================

@st.fragment(
    run_every=(
        st.session_state.replay_speed
        if (
            st.session_state.mode == "Sensor Stream Replay"
            and st.session_state.monitoring
        )
        else None
    )
)
def sensor_replay():

    if st.session_state.mode != "Sensor Stream Replay":

        return


    if not st.session_state.monitoring:

        st.info(
            "Monitoring is paused. "
            "Start the sensor replay to begin."
        )

        return


    # --------------------------------------------------------
    # End of dataset
    # --------------------------------------------------------

    if (
        st.session_state.current_row
        >= len(sensor_data)
    ):

        st.session_state.monitoring = False

        st.success(
            "✅ Sensor replay completed."
        )

        return


    # --------------------------------------------------------
    # Read current sensor record
    # --------------------------------------------------------

    row_number = st.session_state.current_row

    row = sensor_data.iloc[row_number]

    machine_type = str(row["Type"])

    air_temperature = float(
        row["Air temperature [K]"]
    )

    process_temperature = float(
        row["Process temperature [K]"]
    )

    rotational_speed = float(
        row["Rotational speed [rpm]"]
    )

    torque = float(
        row["Torque [Nm]"]
    )

    tool_wear = float(
        row["Tool wear [min]"]
    )


    payload = {
        "Type": machine_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear
    }


    # --------------------------------------------------------
    # API prediction
    # --------------------------------------------------------

    try:

        response = requests.post(
            PREDICT_URL,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        result = response.json()


    except requests.exceptions.Timeout:

        st.error(
            "The predictive-maintenance API "
            "took too long to respond."
        )

        return


    except requests.exceptions.RequestException as e:

        st.error(
            "Unable to connect to the predictive-maintenance API."
        )

        st.caption(
            f"Technical details: {e}"
        )

        return


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    st.session_state.result = result
    st.session_state.last_payload = payload


    # --------------------------------------------------------
    # Failure modes
    # --------------------------------------------------------

    predicted_modes = [
        mode
        for mode, data
        in result["failure_modes"].items()
        if data["predicted"]
    ]


    # --------------------------------------------------------
    # Add history
    # --------------------------------------------------------

    history_record = {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Record": row_number + 1,
        "Type": machine_type,
        "Failure Risk": (
            result["failure_probability"] * 100
        ),
        "Failure": (
            "YES"
            if result["machine_failure"]
            else "NO"
        ),
        "Mode": (
            ", ".join(predicted_modes)
            if predicted_modes
            else "-"
        ),
        "Status": result["maintenance_status"]
    }

    st.session_state.sensor_history.append(
        history_record
    )

    st.session_state.sensor_history = (
        st.session_state.sensor_history[-20:]
    )


    # --------------------------------------------------------
    # Advance to next record
    # --------------------------------------------------------

    st.session_state.current_row += 1


    # ========================================================
    # LIVE MONITORING HEADER
    # ========================================================

    st.success(
        "🟢 LIVE MONITORING — sensor records are being "
        "replayed automatically."
    )

    st.caption(
        f"Record {row_number + 1:,} of {len(sensor_data):,}"
    )


    # ========================================================
    # CURRENT SENSOR READING
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        f'Live Sensor Reading — Record {row_number + 1:,}'
        '</div>',
        unsafe_allow_html=True
    )

    sensor_col1, sensor_col2, sensor_col3 = st.columns(3)


    with sensor_col1:

        st.markdown(
            f"""
            <div class="sensor-card">
                <div class="sensor-label">
                    Machine Type
                </div>
                <div class="sensor-value">
                    {machine_type}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="sensor-card">
                <div class="sensor-label">
                    Air Temperature
                </div>
                <div class="sensor-value">
                    {air_temperature:.1f} K
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with sensor_col2:

        st.markdown(
            f"""
            <div class="sensor-card">
                <div class="sensor-label">
                    Process Temperature
                </div>
                <div class="sensor-value">
                    {process_temperature:.1f} K
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="sensor-card">
                <div class="sensor-label">
                    Rotational Speed
                </div>
                <div class="sensor-value">
                    {rotational_speed:.0f} rpm
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with sensor_col3:

        st.markdown(
            f"""
            <div class="sensor-card">
                <div class="sensor-label">
                    Torque
                </div>
                <div class="sensor-value">
                    {torque:.1f} Nm
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="sensor-card">
                <div class="sensor-label">
                    Tool Wear
                </div>
                <div class="sensor-value">
                    {tool_wear:.0f} min
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # LIVE PREDICTION
    # ========================================================

    display_prediction(result)


    # ========================================================
    # RECENT SENSOR HISTORY
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Recent Sensor Predictions'
        '</div>',
        unsafe_allow_html=True
    )

    history_df = pd.DataFrame(
        st.session_state.sensor_history
    )

    history_df["Failure Risk"] = (
        history_df["Failure Risk"]
        .map(lambda x: f"{x:.2f}%")
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RUN SENSOR REPLAY
# ============================================================

sensor_replay()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        AI4I 2020 Predictive Maintenance System
        · Manual Analysis · Sensor Stream Replay
        · XGBoost · SHAP · FastAPI · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
