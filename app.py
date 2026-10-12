import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping


# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------

st.set_page_config(
    page_title="AI-Powered Energy Forecasting",
    page_icon="⚡",
    layout="wide"
)


# -------------------------------------------------
# APP TITLE
# -------------------------------------------------

st.title("⚡ AI-Powered Energy Forecasting")

st.write(
    "Forecasting Model Development using Machine Learning "
    "and Deep Learning models."
)


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "df" not in st.session_state:
    st.session_state.df = None

if "clean_df" not in st.session_state:
    st.session_state.clean_df = None

if "feature_df" not in st.session_state:
    st.session_state.feature_df = None

if "lr_results" not in st.session_state:
    st.session_state.lr_results = None

if "xgb_results" not in st.session_state:
    st.session_state.xgb_results = None

if "lstm_results" not in st.session_state:
    st.session_state.lstm_results = None

if "gru_results" not in st.session_state:
    st.session_state.gru_results = None


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Dataset",
        "Preprocessing",
        "Feature Engineering",
        "Model Training",
        "Model Evaluation"
    ]
)


# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------

if page == "Home":

    st.header("Project Overview")

    st.write("""
    This application is developed for the ICT Innovation Project.

    The system will:

    - Load electricity consumption data
    - Preprocess time-series data
    - Create time-based and lag features
    - Train Machine Learning models
    - Train Deep Learning models
    - Compare model performance
    - Evaluate models using MAE, RMSE and MAPE
    """)


# -------------------------------------------------
# DATASET PAGE
# -------------------------------------------------

elif page == "Dataset":

    st.header("📂 Dataset")

    uploaded_file = st.file_uploader(
        "Upload Energy Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        st.session_state.df = pd.read_csv(uploaded_file)

        st.session_state.clean_df = None
        st.session_state.feature_df = None
        st.session_state.lr_results = None
        st.session_state.xgb_results = None
        st.session_state.lstm_results = None
        st.session_state.gru_results = None

        st.success("Dataset loaded successfully!")

    if st.session_state.df is not None:

        df = st.session_state.df

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Number of Rows",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Number of Columns",
                df.shape[1]
            )

        st.subheader("Column Names")
        st.write(list(df.columns))

    else:

        st.info("Please upload a CSV energy dataset.")


# -------------------------------------------------
# PREPROCESSING PAGE
# -------------------------------------------------

elif page == "Preprocessing":

    st.header("🧹 Data Preprocessing")

    if st.session_state.df is None:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state.df.copy()

        st.subheader("Original Dataset")

        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())

        if st.button("Run Preprocessing"):

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce"
            )

            df["load"] = pd.to_numeric(
                df["load"],
                errors="coerce"
            )

            df = df.drop_duplicates()
            df = df.sort_values("timestamp")
            df["load"] = df["load"].ffill()
            df = df.dropna()
            df = df.reset_index(drop=True)

            st.session_state.clean_df = df
            st.session_state.feature_df = None
            st.session_state.lr_results = None
            st.session_state.xgb_results = None
            st.session_state.lstm_results = None
            st.session_state.gru_results = None

            st.success(
                "Preprocessing completed successfully!"
            )

        if st.session_state.clean_df is not None:

            clean_df = st.session_state.clean_df

            st.subheader("Cleaned Dataset Preview")
            st.dataframe(clean_df.head())

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Rows After Cleaning",
                    clean_df.shape[0]
                )

            with col2:
                st.metric(
                    "Missing Values",
                    int(clean_df.isnull().sum().sum())
                )


# -------------------------------------------------
# FEATURE ENGINEERING PAGE
# -------------------------------------------------

elif page == "Feature Engineering":

    st.header("⚙️ Feature Engineering")

    if st.session_state.clean_df is None:

        st.warning("Please complete preprocessing first.")

    else:

        df = st.session_state.clean_df.copy()

        st.write(
            "Feature engineering creates additional information "
            "that helps the forecasting models learn electricity "
            "consumption patterns."
        )

        if st.button("Create Features"):

            # Time features
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["month"] = df["timestamp"].dt.month

            df["weekend"] = (
                df["day_of_week"] >= 5
            ).astype(int)

            # Lag features
            df["lag_1"] = df["load"].shift(1)
            df["lag_96"] = df["load"].shift(96)
            df["lag_672"] = df["load"].shift(672)

            df = df.dropna()
            df = df.reset_index(drop=True)

            st.session_state.feature_df = df

            st.session_state.lr_results = None
            st.session_state.xgb_results = None
            st.session_state.lstm_results = None
            st.session_state.gru_results = None

            st.success(
                "Feature engineering completed successfully!"
            )

        if st.session_state.feature_df is not None:

            feature_df = st.session_state.feature_df

            st.subheader("Feature Engineered Dataset")
            st.dataframe(feature_df.head())

            st.subheader("Created Features")

            st.write("""
            **Time Features**
            - hour
            - day_of_week
            - month
            - weekend

            **Lag Features**
            - lag_1 → electricity load 15 minutes ago
            - lag_96 → same time previous day
            - lag_672 → same time previous week
            """)

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Rows",
                    feature_df.shape[0]
                )

            with col2:
                st.metric(
                    "Columns",
                    feature_df.shape[1]
                )

            st.subheader("Column Names")
            st.write(list(feature_df.columns))


# -------------------------------------------------
# MODEL TRAINING PAGE
# -------------------------------------------------

elif page == "Model Training":

    st.header("🤖 Model Training")

    if st.session_state.feature_df is None:

        st.warning(
            "Please complete preprocessing and "
            "feature engineering first."
        )

    else:

        df = st.session_state.feature_df.copy()

        st.success(
            "Feature engineered dataset is ready."
        )

        # -------------------------------------------------
        # SHARED TRAIN / TEST SPLIT
        # -------------------------------------------------

        feature_columns = [
            "hour",
            "day_of_week",
            "month",
            "weekend",
            "lag_1",
            "lag_96",
            "lag_672"
        ]

        X = df[feature_columns]
        y = df["load"]

        split_index = int(
            len(df) * 0.80
        )

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Training Rows",
                len(X_train)
            )

        with col2:
            st.metric(
                "Testing Rows",
                len(X_test)
            )


        # =================================================
        # 1. LINEAR REGRESSION
        # =================================================

        st.divider()

        st.subheader("1. Linear Regression")

        st.write(
            "Linear Regression is used as the baseline forecasting model."
        )

        if st.button("Train Linear Regression"):

            model = LinearRegression()

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            )

            mape = (
                mean_absolute_percentage_error(
                    y_test,
                    predictions
                )
                * 100
            )

            results_df = pd.DataFrame({
                "timestamp":
                    df["timestamp"]
                    .iloc[split_index:]
                    .values,

                "actual_load":
                    y_test.values,

                "predicted_load":
                    predictions
            })

            st.session_state.lr_results = {
                "model": model,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "predictions": results_df
            }

            st.success(
                "Linear Regression training completed!"
            )

        if st.session_state.lr_results is not None:

            results = st.session_state.lr_results

            st.subheader("Linear Regression Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "MAE",
                    f"{results['mae']:.4f}"
                )

            with col2:
                st.metric(
                    "RMSE",
                    f"{results['rmse']:.4f}"
                )

            with col3:
                st.metric(
                    "MAPE",
                    f"{results['mape']:.2f}%"
                )

            st.subheader(
                "Linear Regression: Actual vs Predicted"
            )

            st.dataframe(
                results["predictions"].head(10)
            )

            lr_plot = (
                results["predictions"]
                .head(300)
                .copy()
                .set_index("timestamp")
            )

            st.line_chart(
                lr_plot[
                    [
                        "actual_load",
                        "predicted_load"
                    ]
                ]
            )


        # =================================================
        # 2. XGBOOST
        # =================================================

        st.divider()

        st.subheader("2. XGBoost")

        st.write(
            "XGBoost uses multiple decision trees to learn "
            "more complex patterns in the electricity data."
        )

        if st.button("Train XGBoost"):

            xgb_model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )

            xgb_model.fit(
                X_train,
                y_train
            )

            xgb_predictions = xgb_model.predict(
                X_test
            )

            xgb_mae = mean_absolute_error(
                y_test,
                xgb_predictions
            )

            xgb_rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    xgb_predictions
                )
            )

            xgb_mape = (
                mean_absolute_percentage_error(
                    y_test,
                    xgb_predictions
                )
                * 100
            )

            xgb_results_df = pd.DataFrame({
                "timestamp":
                    df["timestamp"]
                    .iloc[split_index:]
                    .values,

                "actual_load":
                    y_test.values,

                "predicted_load":
                    xgb_predictions
            })

            st.session_state.xgb_results = {
                "model": xgb_model,
                "mae": xgb_mae,
                "rmse": xgb_rmse,
                "mape": xgb_mape,
                "predictions": xgb_results_df
            }

            st.success(
                "XGBoost training completed!"
            )

        if st.session_state.xgb_results is not None:

            results = st.session_state.xgb_results

            st.subheader("XGBoost Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "MAE",
                    f"{results['mae']:.4f}"
                )

            with col2:
                st.metric(
                    "RMSE",
                    f"{results['rmse']:.4f}"
                )

            with col3:
                st.metric(
                    "MAPE",
                    f"{results['mape']:.2f}%"
                )

            st.subheader(
                "XGBoost: Actual vs Predicted"
            )

            st.dataframe(
                results["predictions"].head(10)
            )

            xgb_plot = (
                results["predictions"]
                .head(300)
                .copy()
                .set_index("timestamp")
            )

            st.line_chart(
                xgb_plot[
                    [
                        "actual_load",
                        "predicted_load"
                    ]
                ]
            )


        # -------------------------------------------------
        # PREPARE DEEP LEARNING DATA
        # -------------------------------------------------

        def prepare_sequence_data(dataframe, split_point, sequence_length=96):

            scaler = MinMaxScaler()

            training_load = (
                dataframe["load"]
                .iloc[:split_point]
                .values
                .reshape(-1, 1)
            )

            scaler.fit(
                training_load
            )

            scaled_load = scaler.transform(
                dataframe["load"]
                .values
                .reshape(-1, 1)
            )

            X_sequence_train = []
            y_sequence_train = []

            for i in range(
                sequence_length,
                split_point
            ):

                X_sequence_train.append(
                    scaled_load[
                        i - sequence_length:i
                    ]
                )

                y_sequence_train.append(
                    scaled_load[i]
                )

            X_sequence_test = []
            y_sequence_test = []

            for i in range(
                split_point,
                len(dataframe)
            ):

                X_sequence_test.append(
                    scaled_load[
                        i - sequence_length:i
                    ]
                )

                y_sequence_test.append(
                    scaled_load[i]
                )

            return (
                np.array(X_sequence_train),
                np.array(y_sequence_train),
                np.array(X_sequence_test),
                np.array(y_sequence_test),
                scaler
            )


        # =================================================
        # 3. LSTM
        # =================================================

        st.divider()

        st.subheader("3. LSTM")

        st.write(
            "LSTM is a deep learning sequence model. "
            "It uses the previous 96 readings (24 hours) "
            "to predict the next electricity load."
        )

        if st.button("Train LSTM"):

            with st.spinner(
                "Training LSTM... This may take a few minutes."
            ):

                sequence_length = 96

                (
                    X_lstm_train,
                    y_lstm_train,
                    X_lstm_test,
                    y_lstm_test,
                    lstm_scaler
                ) = prepare_sequence_data(
                    df,
                    split_index,
                    sequence_length
                )

                lstm_model = Sequential([
                    Input(
                        shape=(
                            sequence_length,
                            1
                        )
                    ),

                    LSTM(
                        32
                    ),

                    Dropout(
                        0.20
                    ),

                    Dense(
                        1
                    )
                ])

                lstm_model.compile(
                    optimizer="adam",
                    loss="mse"
                )

                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True
                )

                history = lstm_model.fit(
                    X_lstm_train,
                    y_lstm_train,
                    epochs=8,
                    batch_size=64,
                    validation_split=0.10,
                    shuffle=False,
                    callbacks=[
                        early_stopping
                    ],
                    verbose=0
                )

                lstm_predictions_scaled = (
                    lstm_model.predict(
                        X_lstm_test,
                        verbose=0
                    )
                )

                lstm_predictions = (
                    lstm_scaler.inverse_transform(
                        lstm_predictions_scaled
                    )
                    .flatten()
                )

                actual_lstm = (
                    lstm_scaler.inverse_transform(
                        y_lstm_test
                        .reshape(-1, 1)
                    )
                    .flatten()
                )

                lstm_mae = mean_absolute_error(
                    actual_lstm,
                    lstm_predictions
                )

                lstm_rmse = np.sqrt(
                    mean_squared_error(
                        actual_lstm,
                        lstm_predictions
                    )
                )

                lstm_mape = (
                    mean_absolute_percentage_error(
                        actual_lstm,
                        lstm_predictions
                    )
                    * 100
                )

                lstm_results_df = pd.DataFrame({
                    "timestamp":
                        df["timestamp"]
                        .iloc[split_index:]
                        .values,

                    "actual_load":
                        actual_lstm,

                    "predicted_load":
                        lstm_predictions
                })

                st.session_state.lstm_results = {
                    "model":
                        lstm_model,

                    "mae":
                        lstm_mae,

                    "rmse":
                        lstm_rmse,

                    "mape":
                        lstm_mape,

                    "predictions":
                        lstm_results_df,

                    "epochs":
                        len(
                            history.history["loss"]
                        )
                }

            st.success(
                "LSTM training completed!"
            )

        if st.session_state.lstm_results is not None:

            results = st.session_state.lstm_results

            st.subheader("LSTM Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "MAE",
                    f"{results['mae']:.4f}"
                )

            with col2:
                st.metric(
                    "RMSE",
                    f"{results['rmse']:.4f}"
                )

            with col3:
                st.metric(
                    "MAPE",
                    f"{results['mape']:.2f}%"
                )

            st.write(
                "Epochs completed:",
                results["epochs"]
            )

            st.subheader(
                "LSTM: Actual vs Predicted"
            )

            st.dataframe(
                results["predictions"].head(10)
            )

            lstm_plot = (
                results["predictions"]
                .head(300)
                .copy()
                .set_index("timestamp")
            )

            st.line_chart(
                lstm_plot[
                    [
                        "actual_load",
                        "predicted_load"
                    ]
                ]
            )


        # =================================================
        # 4. GRU
        # =================================================

        st.divider()

        st.subheader("4. GRU")

        st.write(
            "GRU is another deep learning sequence model. "
            "It is similar to LSTM but has a simpler internal structure. "
            "It also uses the previous 96 readings (24 hours) "
            "to predict the next electricity load."
        )

        if st.button("Train GRU"):

            with st.spinner(
                "Training GRU... This may take a few minutes."
            ):

                sequence_length = 96

                (
                    X_gru_train,
                    y_gru_train,
                    X_gru_test,
                    y_gru_test,
                    gru_scaler
                ) = prepare_sequence_data(
                    df,
                    split_index,
                    sequence_length
                )

                gru_model = Sequential([
                    Input(
                        shape=(
                            sequence_length,
                            1
                        )
                    ),

                    GRU(
                        32
                    ),

                    Dropout(
                        0.20
                    ),

                    Dense(
                        1
                    )
                ])

                gru_model.compile(
                    optimizer="adam",
                    loss="mse"
                )

                gru_early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True
                )

                gru_history = gru_model.fit(
                    X_gru_train,
                    y_gru_train,
                    epochs=8,
                    batch_size=64,
                    validation_split=0.10,
                    shuffle=False,
                    callbacks=[
                        gru_early_stopping
                    ],
                    verbose=0
                )

                gru_predictions_scaled = (
                    gru_model.predict(
                        X_gru_test,
                        verbose=0
                    )
                )

                gru_predictions = (
                    gru_scaler.inverse_transform(
                        gru_predictions_scaled
                    )
                    .flatten()
                )

                actual_gru = (
                    gru_scaler.inverse_transform(
                        y_gru_test
                        .reshape(-1, 1)
                    )
                    .flatten()
                )

                gru_mae = mean_absolute_error(
                    actual_gru,
                    gru_predictions
                )

                gru_rmse = np.sqrt(
                    mean_squared_error(
                        actual_gru,
                        gru_predictions
                    )
                )

                gru_mape = (
                    mean_absolute_percentage_error(
                        actual_gru,
                        gru_predictions
                    )
                    * 100
                )

                gru_results_df = pd.DataFrame({
                    "timestamp":
                        df["timestamp"]
                        .iloc[split_index:]
                        .values,

                    "actual_load":
                        actual_gru,

                    "predicted_load":
                        gru_predictions
                })

                st.session_state.gru_results = {
                    "model":
                        gru_model,

                    "mae":
                        gru_mae,

                    "rmse":
                        gru_rmse,

                    "mape":
                        gru_mape,

                    "predictions":
                        gru_results_df,

                    "epochs":
                        len(
                            gru_history.history["loss"]
                        )
                }

            st.success(
                "GRU training completed!"
            )

        if st.session_state.gru_results is not None:

            results = st.session_state.gru_results

            st.subheader("GRU Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "MAE",
                    f"{results['mae']:.4f}"
                )

            with col2:
                st.metric(
                    "RMSE",
                    f"{results['rmse']:.4f}"
                )

            with col3:
                st.metric(
                    "MAPE",
                    f"{results['mape']:.2f}%"
                )

            st.write(
                "Epochs completed:",
                results["epochs"]
            )

            st.subheader(
                "GRU: Actual vs Predicted"
            )

            st.dataframe(
                results["predictions"].head(10)
            )

            gru_plot = (
                results["predictions"]
                .head(300)
                .copy()
                .set_index("timestamp")
            )

            st.line_chart(
                gru_plot[
                    [
                        "actual_load",
                        "predicted_load"
                    ]
                ]
            )


# -------------------------------------------------
# MODEL EVALUATION PAGE
# -------------------------------------------------

elif page == "Model Evaluation":

    st.header("📊 Model Evaluation")

    st.write("""
    Models are evaluated using:

    - MAE — Mean Absolute Error
    - RMSE — Root Mean Squared Error
    - MAPE — Mean Absolute Percentage Error

    Lower values indicate better forecasting performance.
    """)

    if (
        st.session_state.lr_results is None
        and
        st.session_state.xgb_results is None
        and
        st.session_state.lstm_results is None
        and
        st.session_state.gru_results is None
    ):

        st.info(
            "Train at least one model first "
            "to see evaluation results."
        )

    else:

        comparison_rows = []

        if st.session_state.lr_results is not None:

            comparison_rows.append({
                "Model":
                    "Linear Regression",

                "MAE":
                    st.session_state
                    .lr_results["mae"],

                "RMSE":
                    st.session_state
                    .lr_results["rmse"],

                "MAPE (%)":
                    st.session_state
                    .lr_results["mape"]
            })

        if st.session_state.xgb_results is not None:

            comparison_rows.append({
                "Model":
                    "XGBoost",

                "MAE":
                    st.session_state
                    .xgb_results["mae"],

                "RMSE":
                    st.session_state
                    .xgb_results["rmse"],

                "MAPE (%)":
                    st.session_state
                    .xgb_results["mape"]
            })

        if st.session_state.lstm_results is not None:

            comparison_rows.append({
                "Model":
                    "LSTM",

                "MAE":
                    st.session_state
                    .lstm_results["mae"],

                "RMSE":
                    st.session_state
                    .lstm_results["rmse"],

                "MAPE (%)":
                    st.session_state
                    .lstm_results["mape"]
            })

        if st.session_state.gru_results is not None:

            comparison_rows.append({
                "Model":
                    "GRU",

                "MAE":
                    st.session_state
                    .gru_results["mae"],

                "RMSE":
                    st.session_state
                    .gru_results["rmse"],

                "MAPE (%)":
                    st.session_state
                    .gru_results["mape"]
            })


        comparison_df = pd.DataFrame(
            comparison_rows
        )


        st.subheader("Final Model Comparison")

        display_df = comparison_df.copy()

        display_df["MAE"] = (
            display_df["MAE"]
            .round(4)
        )

        display_df["RMSE"] = (
            display_df["RMSE"]
            .round(4)
        )

        display_df["MAPE (%)"] = (
            display_df["MAPE (%)"]
            .round(2)
        )

        st.dataframe(
            display_df,
            use_container_width=True
        )


        if len(comparison_df) > 1:

            best_model = (
                comparison_df
                .sort_values("RMSE")
                .iloc[0]
            )

            st.success(
                f"Best performing model based on RMSE: "
                f"{best_model['Model']} "
                f"(RMSE = {best_model['RMSE']:.4f})"
            )


        st.subheader("MAE Comparison")

        mae_chart = (
            comparison_df[
                [
                    "Model",
                    "MAE"
                ]
            ]
            .set_index("Model")
        )

        st.bar_chart(
            mae_chart
        )


        st.subheader("RMSE Comparison")

        rmse_chart = (
            comparison_df[
                [
                    "Model",
                    "RMSE"
                ]
            ]
            .set_index("Model")
        )

        st.bar_chart(
            rmse_chart
        )


        st.subheader("MAPE Comparison")

        mape_chart = (
            comparison_df[
                [
                    "Model",
                    "MAPE (%)"
                ]
            ]
            .set_index("Model")
        )

        st.bar_chart(
            mape_chart
        )
