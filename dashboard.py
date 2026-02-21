import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from llm_image_agent import generate_image_insights
import plotly.io as pio


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

df = pd.read_sql_table('demo_rawdata1', con=engine)

df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')


# ==============================
# COLOR SYSTEM
# ==============================

PAGE_BG = "#F3F4F6"
CARD_BG = "#FFFFFF"
HEADER_BG = "linear-gradient(90deg,#1E3A8A,#2563EB)"

TOP_COLORS = [
    "#2563EB",
    "#7C3AED",
    "#EC4899",
    "#F97316",
    "#10B981",
    "#6366F1",
    "#14B8A6",
    "#8B5CF6",
    "#E11D48",
    "#22C55E"
]


BUTTON_STYLE = {
    "backgroundColor": "#2563EB",
    "color": "white",
    "border": "none",
    "padding": "10px 20px",
    "borderRadius": "8px",
    "fontWeight": "600",
    "cursor": "pointer",
    "marginTop": "10px",
    "boxShadow": "0 4px 10px rgba(0,0,0,0.15)"
}


INSIGHT_CARD_STYLE = {
    "backgroundColor": "#FFFFFF",
    "padding": "15px",
    "borderRadius": "10px",
    "marginTop": "10px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.1)",
    "whiteSpace": "pre-wrap",
    "fontSize": "15px"
}



# ==============================
# STYLE FUNCTION
# ==============================

def style_fig(fig):
    fig.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color="#111827"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=420
    )
    return fig


# ==============================
# KPI DATA
# ==============================

total_sales = df['Gross_Sales'].sum()
total_profit = df['Profit_Amount'].sum()
total_qty = df['Quantity_Sold'].sum()
avg_margin = df['Profit_Margin_%'].mean()


def kpi_card(value, label):
    return html.Div([
        html.H2(value),
        html.P(label)
    ], style={"backgroundColor": CARD_BG, "padding": "20px"})


# ==============================
# DATA PREP
# ==============================

daily_sales = df.groupby('Date', as_index=False)['Gross_Sales'].sum()
daily_sales['Cumulative'] = daily_sales['Gross_Sales'].cumsum()

category_sales = df.groupby('Product_Category', as_index=False)['Gross_Sales'].sum()
store_sales = df.groupby('Store_Location', as_index=False)['Gross_Sales'].sum()

top_products = df.groupby('Product_Name', as_index=False)['Quantity_Sold'].sum().head(10)

product_perf = df.groupby(
    ['Product_Name', 'Product_Category'],
    as_index=False
).agg({
    'Unit_Price': 'mean',
    'Quantity_Sold': 'sum',
    'Gross_Sales': 'sum'
})


# ==============================
# CHARTS
# ==============================

fig1 = style_fig(px.line(daily_sales, x="Date", y="Gross_Sales"))
fig2 = style_fig(px.bar(category_sales, x="Gross_Sales", y="Product_Category"))
fig3 = style_fig(px.pie(store_sales, names="Store_Location", values="Gross_Sales"))
fig4 = style_fig(px.bar(top_products, x="Quantity_Sold", y="Product_Name"))
fig5 = style_fig(px.line(daily_sales, x="Date", y="Cumulative"))
fig6 = style_fig(px.scatter(product_perf, x="Unit_Price", y="Quantity_Sold"))


# ==============================
# DASH APP
# ==============================

app = Dash(__name__)


app.layout = html.Div([

    html.H1("SALES FORECAST ANALYSIS DASHBOARD (2026)", style={
        "background": HEADER_BG,
        "color": "white",
        "padding": "18px",
        "textAlign": "center",
        "fontSize": "24px",
        "fontWeight": "600",
        "letterSpacing": "1px"
    }),

    html.Div([
        kpi_card(f"{total_sales:,.0f}", "Total Sales"),
        kpi_card(f"{total_profit:,.0f}", "Total Profit"),
        kpi_card(f"{total_qty:,.0f}", "Total Quantity"),
        kpi_card(f"{avg_margin:.1f}%", "Avg Margin")
    ],style={"display": "flex", "justifyContent": "space-between"}),
    

    # Chart 1
html.Div([
    dcc.Graph(id="chart1", figure=fig1),

    html.Div(
        html.Button("Explain Chart", id="btn-chart1", style=BUTTON_STYLE),
        style={"textAlign": "center"}
    ),

    html.Div(id="insights-chart1", style=INSIGHT_CARD_STYLE)

], style={"marginBottom": "30px"}),


# Chart 2
html.Div([
    dcc.Graph(id="chart2", figure=fig2),

    html.Div(
        html.Button("Explain Chart", id="btn-chart2", style=BUTTON_STYLE),
        style={"textAlign": "center"}
    ),

    html.Div(id="insights-chart2", style=INSIGHT_CARD_STYLE)

], style={"marginBottom": "30px"}),


# Chart 3
html.Div([
    dcc.Graph(id="chart3", figure=fig3),

    html.Div(
        html.Button("Explain Chart", id="btn-chart3", style=BUTTON_STYLE),
        style={"textAlign": "center"}
    ),

    html.Div(id="insights-chart3", style=INSIGHT_CARD_STYLE)

], style={"marginBottom": "30px"}),


# Chart 4
html.Div([
    dcc.Graph(id="chart4", figure=fig4),

    html.Div(
        html.Button("Explain Chart", id="btn-chart4", style=BUTTON_STYLE),
        style={"textAlign": "center"}
    ),

    html.Div(id="insights-chart4", style=INSIGHT_CARD_STYLE)

], style={"marginBottom": "30px"}),


# Chart 5
html.Div([
    dcc.Graph(id="chart5", figure=fig5),

    html.Div(
        html.Button("Explain Chart", id="btn-chart5", style=BUTTON_STYLE),
        style={"textAlign": "center"}
    ),

    html.Div(id="insights-chart5", style=INSIGHT_CARD_STYLE)

], style={"marginBottom": "30px"}),


# Chart 6
html.Div([
    dcc.Graph(id="chart6", figure=fig6),

    html.Div(
        html.Button("Explain Chart", id="btn-chart6", style=BUTTON_STYLE),
        style={"textAlign": "center"}
    ),

    html.Div(id="insights-chart6", style=INSIGHT_CARD_STYLE)

], style={"marginBottom": "30px"}),



], style={"backgroundColor": PAGE_BG})


# ==============================
# CALLBACKS (OUTSIDE LAYOUT)
# ==============================

@app.callback(
    Output("insights-chart1", "children"),
    Input("btn-chart1", "n_clicks"),
    prevent_initial_call=True
)
def explain_chart1(n):
    pio.write_image(fig1, "chart1.png")
    insights = generate_image_insights("chart1.png")

    return html.Pre(insights)


@app.callback(
    Output("insights-chart2", "children"),
    Input("btn-chart2", "n_clicks"),
    prevent_initial_call=True
)
def explain_chart2(n):
    pio.write_image(fig2, "chart2.png")
    return generate_image_insights("chart2.png")


@app.callback(
    Output("insights-chart3", "children"),
    Input("btn-chart3", "n_clicks"),
    prevent_initial_call=True
)
def explain_chart3(n):
    pio.write_image(fig3, "chart3.png")
    return generate_image_insights("chart3.png")


@app.callback(
    Output("insights-chart4", "children"),
    Input("btn-chart4", "n_clicks"),
    prevent_initial_call=True
)
def explain_chart4(n):
    pio.write_image(fig4, "chart4.png")
    return generate_image_insights("chart4.png")


@app.callback(
    Output("insights-chart5", "children"),
    Input("btn-chart5", "n_clicks"),
    prevent_initial_call=True
)
def explain_chart5(n):
    pio.write_image(fig5, "chart5.png")
    return generate_image_insights("chart5.png")


@app.callback(
    Output("insights-chart6", "children"),
    Input("btn-chart6", "n_clicks"),
    prevent_initial_call=True
)
def explain_chart6(n):
    pio.write_image(fig6, "chart6.png")
    return generate_image_insights("chart6.png")


# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":
    app.run(debug=False)
