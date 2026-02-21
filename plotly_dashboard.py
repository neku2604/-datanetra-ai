import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ================= DATA PREPROCESS =================
def load_and_preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    return df


# ================= KPI SCORECARDS =================
def get_kpi_scorecards(df_filtered):

    fig = make_subplots(rows=1, cols=4,
                        specs=[[{'type': 'indicator'}]*4])

    metrics = [
        ("Total Sales", df_filtered['Gross_Sales'].sum(), "$", ",.0f"),
        ("Total Profit", df_filtered['Profit_Amount'].sum(), "$", ",.0f"),
        ("Avg Margin", df_filtered['Profit_Margin_%'].mean(), "", ".1f"),
        ("Total Units", df_filtered['Quantity_Sold'].sum(), "", ",")
    ]

    for i, (label, value, prefix, fmt) in enumerate(metrics, 1):

        fig.add_trace(go.Indicator(
            mode="number",
            value=value,
            title={"text": label},
            number={'prefix': prefix, 'valueformat': fmt}
        ), row=1, col=i)

    fig.update_layout(height=200, template="plotly_white")

    return fig


# ================= SALES TREND =================
def get_sales_trend(df_filtered):

    trend = df_filtered.groupby('Date')['Gross_Sales'].sum().reset_index()

    fig = px.line(
        trend,
        x='Date',
        y='Gross_Sales',
        title='Daily Sales Performance',
        template="plotly_white"
    )

    return fig


# ================= CATEGORY PERFORMANCE =================
def get_category_performance(df_filtered):

    cat_data = df_filtered.groupby('Product_Category')['Gross_Sales'].sum().reset_index()

    fig = px.bar(
        cat_data,
        x='Gross_Sales',
        y='Product_Category',
        orientation='h',
        title='Sales by Product Category'
    )

    return fig


# ================= LOCATION CHART =================
def get_location_chart(df_filtered):

    loc_data = df_filtered.groupby('Store_Location')['Gross_Sales'].sum().reset_index()

    fig = px.pie(
        loc_data,
        values='Gross_Sales',
        names='Store_Location',
        title='Revenue Share by Store Location'
    )

    return fig


# ================= TOP PRODUCTS =================
def get_top_products(df_filtered):

    prod_data = df_filtered.groupby('Product_Name')['Gross_Sales'].sum().reset_index()

    fig = px.bar(
        prod_data,
        x='Product_Name',
        y='Gross_Sales',
        title='Top Products by Revenue'
    )

    return fig


# ================= PRICE VS QUANTITY =================
def get_price_vs_quantity_scatter(df_filtered):

    df_agg = df_filtered.groupby(['Product_Name', 'Product_Category']).agg({

        'Unit_Price': 'mean',
        'Quantity_Sold': 'sum',
        'Gross_Sales': 'sum'

    }).reset_index()

    fig = px.scatter(
        df_agg,
        x='Unit_Price',
        y='Quantity_Sold',
        color='Product_Category',
        size='Gross_Sales',
        hover_name='Product_Name',
        title='Price vs Quantity'
    )

    return fig


# ================= CUMULATIVE SALES =================
def get_cumulative_sales_chart(df_filtered):

    daily_sales = df_filtered.groupby('Date')['Gross_Sales'].sum().reset_index()

    daily_sales['Cumulative'] = daily_sales['Gross_Sales'].cumsum()

    fig = px.line(
        daily_sales,
        x='Date',
        y='Cumulative',
        title='Cumulative Sales'
    )

    return fig

# ================= QUANTITY BY PRODUCT =================
def get_quantity_by_product_chart(df_filtered, top_n=10):

    qty_data = (
        df_filtered.groupby('Product_Name')['Quantity_Sold']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig = px.bar(
        qty_data,
        x='Quantity_Sold',
        y='Product_Name',
        orientation='h',
        title='Top Products by Quantity Sold',
        template="plotly_white",
        color='Quantity_Sold'
    )

    fig.update_layout(yaxis={'categoryorder': 'total ascending'})

    return fig
