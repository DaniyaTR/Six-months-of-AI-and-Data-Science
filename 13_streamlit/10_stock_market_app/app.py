# Import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Page settings
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Title
st.title("Stock Market Forecasting App")
st.subheader(
    "This app is created to forecast the stock market price of the selected company."
)

st.image(
    "https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg"
)


# Sidebar
st.sidebar.header("Select the parameters from below")

start_date = st.sidebar.date_input(
    "Start date",
    date(2020, 1, 1)
)

end_date = st.sidebar.date_input(
    "End date",
    date(2020, 12, 31)
)


# Companies
ticker_list = [
    "AAPL",
    "MSFT",
    "GOOG",
    "GOOGL",
    "META",
    "TSLA",
    "NVDA",
    "ADBE",
    "PYPL",
    "INTC",
    "CMCSA",
    "NFLX",
    "PEP"
]

ticker = st.sidebar.selectbox(
    "Select the company",
    ticker_list
)


# Download stock data
data = yf.download(
    ticker,
    start=start_date,
    end=end_date
)

if data.empty:
    st.error("No data found. Please select a different date range.")
    st.stop()


# Handle yfinance multi-level columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)


# Add Date column
data.insert(0, "Date", data.index)
data.reset_index(drop=True, inplace=True)

st.write(
    "Data from",
    start_date,
    "to",
    end_date
)

st.write(data)


# Data visualization
st.header("Data Visualization")
st.subheader("Plot of the data")

fig = px.line(
    data,
    x="Date",
    y=data.columns,
    title="Stock Price",
    width=1000,
    height=600
)

st.plotly_chart(fig)


# Select forecasting column
column = st.selectbox(
    "Select the column to be used for forecasting",
    data.columns[1:]
)


# Selected data
data = data[["Date", column]].copy()

st.write("Selected Data")
st.write(data)


# Make sure price column is numeric
data[column] = pd.to_numeric(
    data[column],
    errors="coerce"
)

data.dropna(
    subset=[column],
    inplace=True
)


# Stationarity test
st.header("Is data Stationary?")

try:
    adf_result = adfuller(data[column])
    st.write(adf_result[1] < 0.05)
except Exception as e:
    st.warning(f"ADF test could not be completed: {e}")


# Decomposition
st.header("Decomposition of the data")

try:
    decomposition = seasonal_decompose(
        data[column],
        model="additive",
        period=12
    )

    st.pyplot(decomposition.plot())

except Exception as e:
    st.warning(
        f"Decomposition could not be completed: {e}"
    )


# Model selection
models = [
    "SARIMA",
    "Random Forest",
    "Prophet"
]

selected_model = st.sidebar.selectbox(
    "Select the model for forecasting",
    models
)


# ============================================================
# SARIMA
# ============================================================

if selected_model == "SARIMA":

    st.header("SARIMA Model")

    p = st.slider(
        "Select the value of p",
        0,
        5,
        2
    )

    d = st.slider(
        "Select the value of d",
        0,
        5,
        1
    )

    q = st.slider(
        "Select the value of q",
        0,
        5,
        2
    )

    seasonal_period = st.number_input(
        "Select the seasonal period",
        1,
        24,
        12
    )

    if st.button("Run SARIMA Forecast"):

        try:

            model = sm.tsa.statespace.SARIMAX(
                data[column],
                order=(p, d, q),
                seasonal_order=(
                    p,
                    d,
                    q,
                    seasonal_period
                )
            )

            model = model.fit(
                disp=False
            )

            st.header("Model Summary")
            st.write(model.summary())

            forecast_period = st.number_input(
                "Select the number of days to forecast",
                1,
                365,
                10
            )

            predictions = model.get_prediction(
                start=len(data),
                end=len(data) + forecast_period - 1
            )

            predictions = predictions.predicted_mean

            future_dates = pd.date_range(
                start=data["Date"].iloc[-1] + pd.Timedelta(days=1),
                periods=forecast_period,
                freq="D"
            )

            predictions_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted": predictions.values
            })

            st.write(
                "Predictions",
                predictions_df
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data[column],
                    mode="lines",
                    name="Actual"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=predictions_df["Date"],
                    y=predictions_df["Predicted"],
                    mode="lines",
                    name="Predicted"
                )
            )

            fig.update_layout(
                title="Actual vs Predicted - SARIMA",
                xaxis_title="Date",
                yaxis_title="Price",
                width=1000,
                height=400
            )

            st.plotly_chart(fig)

        except Exception as e:
            st.error(
                f"SARIMA error: {e}"
            )


# ============================================================
# RANDOM FOREST
# ============================================================

elif selected_model == "Random Forest":

    st.header("Random Forest Regression")

    train_size = int(
        len(data) * 0.8
    )

    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    if len(test_data) == 0:
        st.error(
            "Not enough data for training and testing."
        )
        st.stop()

    train_X = (
        train_data["Date"]
        .map(pd.Timestamp.toordinal)
        .values
        .reshape(-1, 1)
    )

    train_y = train_data[column].values

    test_X = (
        test_data["Date"]
        .map(pd.Timestamp.toordinal)
        .values
        .reshape(-1, 1)
    )

    test_y = test_data[column].values

    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=0
    )

    rf_model.fit(
        train_X,
        train_y
    )

    predictions = rf_model.predict(
        test_X
    )

    mse = mean_squared_error(
        test_y,
        predictions
    )

    rmse = np.sqrt(mse)

    st.write(
        f"Root Mean Squared Error (RMSE): {rmse}"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data[column],
            mode="lines",
            name="Actual"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_data["Date"],
            y=predictions,
            mode="lines",
            name="Predicted"
        )
    )

    fig.update_layout(
        title="Actual vs Predicted - Random Forest",
        xaxis_title="Date",
        yaxis_title="Price",
        width=1000,
        height=400
    )

    st.plotly_chart(fig)


# ============================================================
# PROPHET
# ============================================================

elif selected_model == "Prophet":

    st.header("Facebook Prophet")

    try:

        prophet_data = data[
            ["Date", column]
        ].copy()

        prophet_data.rename(
            columns={
                "Date": "ds",
                column: "y"
            },
            inplace=True
        )

        prophet_data["y"] = pd.to_numeric(
            prophet_data["y"],
            errors="coerce"
        )

        prophet_data.dropna(
            inplace=True
        )

        prophet_model = Prophet()

        prophet_model.fit(
            prophet_data
        )

        future = prophet_model.make_future_dataframe(
            periods=365
        )

        forecast = prophet_model.predict(
            future
        )

        st.write(
            forecast[
                ["ds", "yhat", "yhat_lower", "yhat_upper"]
            ].tail(365)
        )

        fig = prophet_model.plot(
            forecast
        )

        plt.title(
            "Forecast with Facebook Prophet"
        )

        plt.xlabel("Date")
        plt.ylabel("Price")

        st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Prophet error: {e}"
        )


# Footer
st.write(
    "Model selected:",
    selected_model
)