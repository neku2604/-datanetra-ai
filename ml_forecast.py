import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

def forecast_sales(csv_file_path):
    """
    Develops a sales forecasting function that encapsulates the entire sales forecasting process.

    Args:
        csv_file_path (str): The path to the CSV file containing sales data.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object visualizing the historical
                                     monthly sales and the 1-year forecasted monthly sales.
    """

    # 1. Load the data
    df = pd.read_csv(csv_file_path)

    # 2. Preprocess data to create monthly aggregated sales
    df_model = df.copy()
    df_model['Date'] = pd.to_datetime(df_model['Date'], format='%Y-%m-%d')
    df_model = df_model[['Date', 'Gross_Sales']]
    df_model = df_model.set_index('Date')
    monthly_sales = df_model.resample('MS').sum().reset_index()
    monthly_sales = monthly_sales.rename(columns={'Date': 'ds', 'Gross_Sales': 'y'})
    monthly_sales['ds'] = pd.to_datetime(monthly_sales['ds'])
    monthly_sales = monthly_sales.sort_values('ds')

    # 3. Create train_df with the last 12 months of data
    ROLLING_MONTHS = 12
    if len(monthly_sales) > ROLLING_MONTHS:
        train_df = monthly_sales.tail(ROLLING_MONTHS)
    else:
        train_df = monthly_sales.copy()

    # 4. Initialize Prophet model with specified parameters
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.001,
        seasonality_prior_scale=25,
        changepoint_range=0.95,
        interval_width=0.95
    )

    # 5. Train the Prophet model
    model.fit(train_df)

    # 6. Generate future dataframe for a 1-year (12 months) forecast
    future = model.make_future_dataframe(periods=12, freq='MS')

    # 7. Predict sales for the future period
    forecast = model.predict(future)
        

    # Keep only future dates for plotting forecast
    last_train_date = monthly_sales['ds'].max()
    forecast_future = forecast[forecast['ds'] >= last_train_date].copy()
    forecast_future = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # 8. Create a Plotly graph
    fig = go.Figure()

    # Historical sales trace
    fig.add_trace(go.Scatter(
        x=monthly_sales['ds'],
        y=monthly_sales['y'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='#1f77b4')
    ))

    # 1-Year Forecast trace
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat'],
        mode='lines',
        name='1-Year Forecast',
        line=dict(color='#003366', dash='dash')
    ))

    # Confidence interval shading (optional, as commented out in original notebook)
    # fig.add_trace(go.Scatter(
    #     x=forecast_future['ds'],
    #     y=forecast_future['yhat_upper'],
    #     fill=None,
    #     mode='lines',
    #     line=dict(color='lightblue'),
    #     showlegend=False
    # ))

    # fig.add_trace(go.Scatter(
    #     x=forecast_future['ds'],
    #     y=forecast_future['yhat_lower'],
    #     fill='tonexty',
    #     mode='lines',
    #     line=dict(color='lightblue'),
    #     name='Confidence Interval'
    # ))

    # Update layout
    fig.update_layout(
        title="ML Forecast (1 Year)",
        xaxis_title="Month",
        yaxis_title="Sales",
        template="plotly_white"
    )

    # 9. Return the Plotly figure object
    return fig

print("forecast_sales function defined.")