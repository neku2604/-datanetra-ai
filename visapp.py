
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sqlalchemy import create_engine

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from ml_forecast import forecast_sales

# IMPORT YOUR LLM IMAGE FUNCTION
from llm_image_agent import generate_image_insights

def format_indian_currency(value):
    value = round(value)

    if value >= 10000000:   # Crore
        return f"₹{value/10000000:.1f} Cr"
    elif value >= 100000:   # Lakh
        return f"₹{value/100000:.1f} L"
    elif value >= 1000:     # Thousand
        return f"₹{value/1000:.1f} K"
    else:
        return f"₹{value}"


# ==============================
# DATABASE CONNECTION
# ==============================

host = "aws-1-ap-southeast-1.pooler.supabase.com"
port = "6543"
database = "postgres"
user = "postgres.lmkavbsqutyrshafjvak"
password = "DataNetra123!"

engine = create_engine(
    f"postgresql://{user}:{password}@{host}:{port}/{database}"
)

df = pd.read_sql_table("demo_rawdata1", con=engine)

df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")


# ==============================
# KPI DATA
# ==============================

total_sales = df["Gross_Sales"].sum()
total_profit = df["Profit_Amount"].sum()
total_qty = df["Quantity_Sold"].sum()
avg_margin = df["Profit_Margin_%"].mean()


def kpi_card(value, label):

    return html.Div([
        html.H2(value),
        html.P(label)
    ],
    style={
        "backgroundColor": "white",
        "padding": "20px",
        "margin": "10px",
        "borderRadius": "10px",
        "boxShadow": "0px 4px 12px rgba(0,0,0,0.15)",
        "textAlign": "center",
        "flex": "1"
    })


# ==============================
# PREPARE DATA
# ==============================

daily_sales = df.groupby("Date", as_index=False)["Gross_Sales"].sum()

category_sales = df.groupby(
    "Product_Category", as_index=False
)["Gross_Sales"].sum()

store_sales = df.groupby(
    "Store_Location", as_index=False
)["Gross_Sales"].sum()

top_products = df.groupby(
    "Product_Name", as_index=False
)["Quantity_Sold"].sum().sort_values(
    by="Quantity_Sold",
    ascending=False
).head(10)


# ==============================
# CREATE CHARTS
# ==============================

# --------------------------------------------

# fig1 = px.line(
#     daily_sales,
#     x="Date",
#     y="Gross_Sales",
#     title="Daily Sales Performance"
# )
# ----------------------------------------------

daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])

# 2. Resample the data to a Monthly level
# 'MS' stands for Month Start (e.g., 2023-01-01, 2023-02-01)
# We sum the 'Gross_Sales' for each month
monthly_sales = daily_sales.resample('MS', on='Date').sum().reset_index()

# 3. Plot the new monthly DataFrame
fig1 = px.line(
    monthly_sales,
    x="Date",
    y="Gross_Sales",
    title="Monthly Sales Performance"
)

# fig1.show()

fig2 = px.bar(
    category_sales,
    x="Gross_Sales",
    y="Product_Category",
    orientation="h",
    title="Sales by Category"
)

fig3 = px.pie(
    store_sales,
    names="Store_Location",
    values="Gross_Sales",
    title="Sales by Store"
)

fig4 = px.bar(
    top_products,
    x="Quantity_Sold",
    y="Product_Name",
    orientation="h",
    title="Top Products"
)

# ==============================
# ML FORECAST CHART (PROPHET)
# ==============================

csv_path = "uploaded_data.csv"

try:
    fig5 = forecast_sales(csv_path)
except Exception as e:
    print("Forecast error:", e)
    fig5 = px.line(title="Forecast unavailable")


# ==============================
# DASH APP
# ==============================

app = Dash(__name__)


# ==============================
# CHART BLOCK WITH BUTTON
# ==============================

def chart_block(chart_id, button_id, insight_id, fig):

    return html.Div([

        dcc.Graph(
            id=chart_id,
            figure=fig
        ),

        html.Button(
            "Explain Chart",
            id=button_id,
            n_clicks=0,
            style={
                 "backgroundColor": "#2563EB",
                  
                    "color": "white",
                    "border": "none",
                    "padding": "10px 18px",
                    "marginTop": "8px",
                    "marginBottom": "12px",
                    "borderRadius": "6px",
                    "fontWeight": "600",
                    "cursor": "pointer"
        
            }
        ),

        html.Div(
            id=insight_id,
            style={
                 "whiteSpace": "pre-wrap",

        "backgroundColor": "#FFFFFF",

        "color": "#1F2937",

        "padding": "18px",

        "borderRadius": "10px",

        "marginBottom": "30px",

        "border": "1px solid #E5E7EB",

        "boxShadow": "0px 4px 12px rgba(0,0,0,0.08)",

        "fontSize": "15px",

        "lineHeight": "1.6",

        "fontFamily": "Segoe UI"
            }
        )

    ])



# ==============================
# LAYOUT
# ==============================

app.layout = html.Div([
html.Button(
    "← Back",
    id="back-btn",
    n_clicks=0,
    style={
        "backgroundColor": "#2563EB",
        "color": "white",
        "border": "none",
        "padding": "8px 16px",
        "borderRadius": "6px",
        "cursor": "pointer",
        "fontWeight": "600",
        "position": "absolute",
        "top": "20px",
        "left": "20px",
        "zIndex": "999"
    }
),
    html.H1(
        "Sales Forecast Analysis Dashboard",
        style={"textAlign": "center"}
    ),

   html.Div([
    kpi_card(format_indian_currency(total_sales), "Total Sales"),
    kpi_card(format_indian_currency(total_profit), "Total Profit"),
    kpi_card(f"{round(total_qty):,}", "Quantity Sold"),
    kpi_card(f"{avg_margin:.0f}%", "Avg Margin"),
],
    style={"display": "flex"}),

    chart_block("chart1", "btn1", "insight1", fig1),
    chart_block("chart2", "btn2", "insight2", fig2),
    chart_block("chart3", "btn3", "insight3", fig3),
    chart_block("chart4", "btn4", "insight4", fig4),
    chart_block("chart5", "btn5", "insight5", fig5),   # NEW ML FORECAST

],
style={"padding": "20px", "backgroundColor": "#F3F4F6"})


# ==============================
# FUNCTION TO PROCESS CHART
# ==============================

def explain_chart(n_clicks, figure, filename):

    if n_clicks == 0:
        return ""

    try:

        image_path = f"{filename}.png"

        pio.write_image(
            figure,
            image_path,
            width=1000,
            height=600
        )

        insights = generate_image_insights(image_path)

        # Remove markdown bold stars
        cleaned = insights.replace("**", "")

        return cleaned

    except Exception as e:

        return f"Error: {str(e)}"


# ==============================
# CALLBACKS
# ==============================

@app.callback(
    Output("insight1", "children"),
    Input("btn1", "n_clicks"),
    State("chart1", "figure")
)
def explain1(n, fig):
    return explain_chart(n, fig, "chart1")


@app.callback(
    Output("insight2", "children"),
    Input("btn2", "n_clicks"),
    State("chart2", "figure")
)
def explain2(n, fig):
    return explain_chart(n, fig, "chart2")


@app.callback(
    Output("insight3", "children"),
    Input("btn3", "n_clicks"),
    State("chart3", "figure")
)
def explain3(n, fig):
    return explain_chart(n, fig, "chart3")


@app.callback(
    Output("insight4", "children"),
    Input("btn4", "n_clicks"),
    State("chart4", "figure")
)
def explain4(n, fig):
    return explain_chart(n, fig, "chart4")

@app.callback(
    Output("insight5", "children"),
    Input("btn5", "n_clicks"),
    State("chart5", "figure")
)

def explain5(n, fig):
    return explain_chart(n, fig, "chart5")

app.clientside_callback(
    """
    function(n_clicks){
        if(n_clicks > 0){
            window.close();
        }
        return "";
    }
    """,
    Output("back-btn", "title"),
    Input("back-btn", "n_clicks")
)


# ==============================
# RUN FUNCTION
# ==============================

def run_dash():

    app.run(
        host="127.0.0.1",
        port=8050,
        debug=False
    )