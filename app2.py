
import gradio as gr
import pandas as pd
import numpy as np
from io import BytesIO
import pdfplumber
import matplotlib.pyplot as plt
import datetime
import os
from llm_agent import generate_insights as generate_llm_insights

import warnings
import threading
import webbrowser
from visapp import run_dash
import time # Import the time module
warnings.filterwarnings('ignore')
from sqlalchemy import create_engine



# ==================== SUPABASE DATABASE SETUP ====================
# SQLAlchemy imports (REQUIRED)
from sqlalchemy import create_engine, Column, BigInteger, String, Float, DateTime, Boolean, Text,text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


SUPABASE_HOST = "aws-1-ap-southeast-1.pooler.supabase.com"
SUPABASE_PORT = "6543"
SUPABASE_DB = "postgres"
SUPABASE_USER = "postgres.lmkavbsqutyrshafjvak"
SUPABASE_PASSWORD = "DataNetra123!"

DATABASE_URL = f"postgresql://{SUPABASE_USER}:{SUPABASE_PASSWORD}@{SUPABASE_HOST}:{SUPABASE_PORT}/{SUPABASE_DB}"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

print("✅ Gradio connected to Supabase")



# ML Models
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not installed. Using fallback forecasting.")

# ==================== DATABASE SETUP ====================

from sqlalchemy import Column, BigInteger, String

class PlatformUser(Base):
    __tablename__ = "platform_user"

    user_id = Column(BigInteger, primary_key=True, index=True)
    full_name = Column(String)
    mobile_number = Column(BigInteger, unique=True, index=True)
    email = Column(String)
    role = Column(String)
    msme_number = Column(String)
    company_name = Column(String)
    organisation_type = Column(String)
    major_activity = Column(String)
    enterprise_type = Column(String)
    state = Column(String)
    city = Column(String)
    business_type = Column(String)
    years_operation = Column(BigInteger)
    monthly_revenue_range = Column(String)
    consent_given = Column(Boolean)


class UdyamMaster(Base):
    __tablename__ = "udyam_master"

    msme_number = Column(String, primary_key=True)
    enterprise_name = Column(String)
    organization_type = Column(String)
    major_activity = Column(String)
    enterprise_type = Column(String)
    state = Column(String)
    city = Column(String)
    district = Column(String)
    pin_code = Column(BigInteger)
    official_mobile = Column(BigInteger)
    official_email = Column(String)
    nic_codes = Column(BigInteger)
    nic_description = Column(String)
    date_of_registration = Column(String)

# Ensure tables exist
Base.metadata.create_all(bind=engine)


# ==================== DATABASE OPERATIONS ====================

def get_user_profile(mobile_number):
    db = SessionLocal()
    try:
        profile = db.query(PlatformUser).filter(
            PlatformUser.mobile_number == int(mobile_number)
        ).first()

        if profile:
            return {
                'user_id': profile.user_id,
                'mobile_number': profile.mobile_number,
                'full_name': profile.full_name,
                'email': profile.email,
                'role': profile.role,
                'msme_number': profile.msme_number
            }
        return None
    finally:
        db.close()

def save_user_profile(profile_data):
    db = SessionLocal()
    try:
        print("DATA RECEIVED:", profile_data)

        user = PlatformUser(
            full_name=profile_data.get("full_name"),
            mobile_number=int(profile_data.get("mobile_number")),
            email=profile_data.get("email"),
            role=profile_data.get("role"),
            msme_number=profile_data.get("msme_number"),
            company_name=profile_data.get("company_name"),
            organisation_type=profile_data.get("organisation_type"),
            major_activity=profile_data.get("major_activity"),
            enterprise_type=profile_data.get("enterprise_type"),
            state=profile_data.get("state"),
            city=profile_data.get("city"),
            business_type=profile_data.get("business_type"),
            years_operation=profile_data.get("years_operation"),
            monthly_revenue_range=profile_data.get("monthly_revenue_range"),
            consent_given=profile_data.get("consent_given")
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        print("SAVED SUCCESSFULLY:", user.user_id)
        return user.user_id

    except Exception as e:
        db.rollback()
        print("DATABASE ERROR:", e)
        raise e
    finally:
        db.close()



# ==================== ML & ANALYTICS FUNCTIONS ====================

def normalize(series):
    """Normalize series to 0-1 range"""
    if series.empty or series.max() == series.min():
        return pd.Series(0, index=series.index)
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

def calculate_scores(df):
    """Calculate risk and performance scores"""
    # Ensure numeric columns
    numeric_cols = ['Monthly_Sales_INR', 'Monthly_Operating_Cost_INR', 'Outstanding_Loan_INR',
                   'Vendor_Delivery_Reliability', 'Inventory_Turnover', 'Avg_Margin_Percent',
                   'Monthly_Demand_Units']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Avoid division by zero
    df['Monthly_Sales_INR'] = df['Monthly_Sales_INR'].replace(0, 1)

    # Calculate scores
    df["Cashflow_Stress"] = normalize(df["Monthly_Operating_Cost_INR"] / df["Monthly_Sales_INR"])
    df["Loan_Stress"] = normalize(df["Outstanding_Loan_INR"] / (df["Monthly_Sales_INR"] * 12))
    df["Financial_Risk_Score"] = (0.5 * df["Cashflow_Stress"] + 0.5 * df["Loan_Stress"]).clip(0, 1)

    df["Vendor_Score"] = (
        0.5 * df["Vendor_Delivery_Reliability"] +
        0.3 * normalize(df["Inventory_Turnover"]) +
        0.2 * normalize(df["Avg_Margin_Percent"])
    ).clip(0, 1)

    df["Growth_Potential_Score"] = (
        0.4 * normalize(df["Monthly_Demand_Units"]) +
        0.35 * normalize(df["Avg_Margin_Percent"]) +
        0.25 * normalize(df.get("Digital_Ad_Spend_INR", pd.Series(0)))
    ).clip(0, 1)

    df["MSME_Health_Score"] = (
        (1 - df["Financial_Risk_Score"]) * 0.4 +
        df["Vendor_Score"] * 0.3 +
        df["Growth_Potential_Score"] * 0.3
    ) * 100

    return df

def forecast_sales(df):
    """Sales forecasting using Prophet or fallback"""
    try:
        if PROPHET_AVAILABLE and 'Date' in df.columns:
            ts_data = df.groupby('Date')['Monthly_Sales_INR'].sum().reset_index()
            ts_data.columns = ['ds', 'y']
            ts_data['ds'] = pd.to_datetime(ts_data['ds'])

            if len(ts_data) < 2:
                raise ValueError("Insufficient data for Prophet")

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(ts_data)

            future_6m = model.make_future_dataframe(periods=180)
            forecast_6m = model.predict(future_6m)

            future_12m = model.make_future_dataframe(periods=365)
            forecast_12m = model.predict(future_12m)

            return {
                '6_month': {
                    'forecast': forecast_6m['yhat'].tail(180).sum(),
                    'lower': forecast_6m['yhat_lower'].tail(180).sum(),
                    'upper': forecast_6m['yhat_upper'].tail(180).sum()
                },
                '12_month': {
                    'forecast': forecast_12m['yhat'].tail(365).sum(),
                    'lower': forecast_12m['yhat_lower'].tail(365).sum(),
                    'upper': forecast_12m['yhat_upper'].tail(365).sum()
                }
            }
        else:
            raise ValueError("Using fallback forecasting")

    except Exception as e:
        print(f"Prophet forecasting failed: {str(e)}. Using fallback.")
        total_sales = df['Monthly_Sales_INR'].sum()
        avg_growth = 0.05

        forecast_6m = total_sales * 6 * (1 + avg_growth)
        forecast_12m = total_sales * 12 * (1 + avg_growth)

        return {
            '6_month': {
                'forecast': forecast_6m,
                'lower': forecast_6m * 0.85,
                'upper': forecast_6m * 1.15
            },
            '12_month': {
                'forecast': forecast_12m,
                'lower': forecast_12m * 0.85,
                'upper': forecast_12m * 1.15
            }
        }

def segment_customers(df):
    """K-Means customer segmentation"""
    try:
        if 'SKU_Name' not in df.columns:
            return None

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            reference_date = df['Date'].max()

            rfm = df.groupby('SKU_Name').agg({
                'Date': lambda x: (reference_date - x.max()).days,
                'Monthly_Sales_INR': ['count', 'sum']
            })

            rfm.columns = ['recency', 'frequency', 'monetary']

            if len(rfm) >= 3:
                scaler = StandardScaler()
                rfm_scaled = scaler.fit_transform(rfm)

                n_clusters = min(5, len(rfm))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                rfm['segment'] = kmeans.fit_predict(rfm_scaled)

                segment_names = ['Champions', 'Loyal', 'Potential', 'At Risk', 'Lost']
                rfm['segment_name'] = rfm['segment'].apply(
                    lambda x: segment_names[x] if x < len(segment_names) else f'Segment {x}'
                )

                return rfm['segment_name'].value_counts().to_dict()

        return None
    except Exception as e:
        print(f"Segmentation error: {str(e)}")
        return None

def generate_insights(user_data, df):
    """Generate AI-powered insights"""
    try:
        company_name = user_data.get('company_name', 'Your Company')
        df = calculate_scores(df)

        # Calculate key metrics
        total_sales = df['Monthly_Sales_INR'].sum()
        total_products = len(df)
        avg_margin = df['Avg_Margin_Percent'].mean()

        top_skus = df.nlargest(5, 'Monthly_Sales_INR')[['SKU_Name', 'Monthly_Sales_INR', 'Monthly_Demand_Units']]

        insights = f"""# 🎯 AI-Powered Business Insights for {company_name}

## 📊 Overall Performance Summary
- **Total Sales:** ₹{total_sales:,.0f}
- **Total Products Analyzed:** {total_products}
- **Average Profit Margin:** {avg_margin:.1f}%
- **Overall MSME Health Score:** {df["MSME_Health_Score"].mean():.1f}%

## 🏆 Top 5 Performing Products
"""
        for idx, row in top_skus.iterrows():
            insights += f"\n{idx+1}. **{row['SKU_Name']}** - ₹{row['Monthly_Sales_INR']:,.0f} | {row['Monthly_Demand_Units']:.0f} units"

        # Performance scores
        insights += f"""

## 📈 Performance Metrics
- **Financial Risk Score:** {df["Financial_Risk_Score"].mean():.2f} (Lower is better)
- **Vendor Reliability Score:** {df["Vendor_Score"].mean():.2f}
- **Growth Potential Score:** {df["Growth_Potential_Score"].mean():.2f}
"""

        # Forecasting
        forecast_results = forecast_sales(df)

        insights += f"""

## 🔮 ML-Powered Sales Forecast
### 6-Month Projection
- **Forecasted Sales:** ₹{forecast_results['6_month']['forecast']:,.0f}
- **Expected Range:** ₹{forecast_results['6_month']['lower']:,.0f} - ₹{forecast_results['6_month']['upper']:,.0f}

### 12-Month Projection
- **Forecasted Sales:** ₹{forecast_results['12_month']['forecast']:,.0f}
- **Expected Range:** ₹{forecast_results['12_month']['lower']:,.0f} - ₹{forecast_results['12_month']['upper']:,.0f}

## 💡 AI-Generated Recommendations

### 🎯 Immediate Actions
1. **Prioritize Top Performers:** Focus inventory and marketing on your top 5 SKUs
2. **Risk Mitigation:** Review products with Financial Risk Score > 0.7
3. **Vendor Management:** Strengthen partnerships with high-reliability vendors

### 📊 Strategic Initiatives
4. **Demand Forecasting:** Use ML predictions to optimize stock levels and reduce waste
5. **Margin Optimization:** Analyze low-margin products for pricing or cost reduction opportunities
6. **Growth Planning:** Invest in products showing high growth potential scores
"""

        # Customer segmentation
        segments = segment_customers(df)
        if segments:
            insights += "\n\n## 👥 Customer Segments (K-Means ML Clustering)\n"
            for segment, count in segments.items():
                insights += f"- **{segment}:** {count} products\n"

        # Risk alerts
        high_risk = df[df['Financial_Risk_Score'] > 0.7]
        if len(high_risk) > 0:
            insights += f"\n\n## ⚠️ Risk Alerts\n{len(high_risk)} products require immediate attention due to high financial risk.\n"

        return insights, None, forecast_results

    except Exception as e:
        error_msg = f"Error generating insights: {str(e)}\n\nPlease ensure your Excel file has the required columns."
        print(f"Insights generation error: {str(e)}")
        return None, error_msg, None

def generate_dashboard_data(user_data, df):
    """Generate dashboard KPIs and charts"""
    try:
        df = calculate_scores(df)

        total_sales = df['Monthly_Sales_INR'].sum()
        avg_margin = df['Avg_Margin_Percent'].mean() if 'Avg_Margin_Percent' in df.columns else 0
        total_profit = total_sales * (avg_margin / 100)
        health_score = df['MSME_Health_Score'].mean()
        growth_score = df['Growth_Potential_Score'].mean()

        # Chart 1: Top SKUs Bar Chart
        plt.style.use('seaborn-v0_8-darkgrid')
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        top_skus = df.nlargest(10, 'Monthly_Sales_INR')
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_skus)))
        bars = ax1.barh(top_skus['SKU_Name'], top_skus['Monthly_Sales_INR'], color=colors)
        ax1.set_xlabel('Sales (INR)', fontsize=12, fontweight='bold')
        ax1.set_title('Top 10 Products by Sales Revenue', fontsize=14, fontweight='bold', pad=20)
        ax1.grid(axis='x', alpha=0.3)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, f'₹{width:,.0f}',
                    ha='left', va='center', fontsize=9, fontweight='bold')
        plt.tight_layout()

        # Chart 2: Score Distribution
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        scores = ['Financial\nRisk', 'Vendor\nScore', 'Growth\nPotential']
        values = [
            df['Financial_Risk_Score'].mean(),
            df['Vendor_Score'].mean(),
            df['Growth_Potential_Score'].mean()
        ]
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        bars = ax2.bar(scores, values, color=colors, alpha=0.8, width=0.6)
        ax2.set_ylabel('Score (0-1)', fontsize=12, fontweight='bold')
        ax2.set_title('Performance Scores Overview', fontsize=14, fontweight='bold', pad=20)
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        plt.tight_layout()

        # Chart 3: Sales vs Margin Scatter
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        scatter = ax3.scatter(df['Monthly_Sales_INR'], df['Avg_Margin_Percent'],
                   alpha=0.6, c=df['MSME_Health_Score'], cmap='RdYlGn',
                   s=100, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Monthly Sales (INR)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Profit Margin (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Sales vs Margin Analysis (Color = Health Score)', fontsize=14, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Health Score', fontsize=11, fontweight='bold')
        plt.tight_layout()

        # Chart 4: Forecast Visualization
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        forecast_results = forecast_sales(df)
        months = ['6-Month\nForecast', '12-Month\nForecast']
        forecasts = [
            forecast_results['6_month']['forecast'],
            forecast_results['12_month']['forecast']
        ]
        lowers = [
            forecast_results['6_month']['lower'],
            forecast_results['12_month']['lower']
        ]
        uppers = [
            forecast_results['6_month']['upper'],
            forecast_results['12_month']['upper']
        ]

        x_pos = np.arange(len(months))
        bars = ax4.bar(x_pos, forecasts, color=['#9b59b6', '#e67e22'], alpha=0.8, width=0.5)

        # Add error bars for confidence intervals
        errors = [[forecasts[i] - lowers[i] for i in range(len(forecasts))],
                 [uppers[i] - forecasts[i] for i in range(len(forecasts))]]
        ax4.errorbar(x_pos, forecasts, yerr=errors, fmt='none', ecolor='black',
                    capsize=5, capthick=2, alpha=0.5)

        ax4.set_ylabel('Forecasted Sales (INR)', fontsize=12, fontweight='bold')
        ax4.set_title('ML Sales Forecast with Confidence Intervals', fontsize=14, fontweight='bold', pad=20)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(months)
        ax4.grid(axis='y', alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, forecasts)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'₹{val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()

        return (
            f"### 💰 Total Sales\n# ₹{total_sales:,.0f}",
            f"### 📈 Total Profit\n# ₹{total_profit:,.0f}",
            f"### 🧠 Health Score\n# {health_score:.1f}%",
            f"### 🚀 Growth Score\n# {growth_score:.2f}",
            fig1, fig2, fig3, fig4,
            None
        )

    except Exception as e:
        print(f"Dashboard generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return ("N/A", "N/A", "N/A", "N/A", None, None, None, None, f"Error: {str(e)}")

def generate_pdf_report(user_data, df, dashboard_figs=None):
    """Generate PDF report"""
    try:
        from matplotlib.backends.backend_pdf import PdfPages

        df = calculate_scores(df)
        forecast_results = forecast_sales(df)

        # pdf_path = os.path.join(os.getcwd(), "report.pdf").replace(" ", "_")
        company = user_data.get("company_name", "Company").replace(" ", "_")
        pdf_path = os.path.join(os.getcwd(), f"{company}_report.pdf")

        with PdfPages(pdf_path) as pdf:
            # Cover page
            fig_cover = plt.figure(figsize=(8, 6))
            plt.axis('off')
            plt.text(0.5, 0.6, "MSME Analytics Report", fontsize=24, ha='center', weight='bold')
            plt.text(0.5, 0.5, f"{user_data.get('company_name', 'Your Company')}", fontsize=16, ha='center')
            plt.text(0.5, 0.4, f"{datetime.datetime.now().strftime('%d %B %Y')}", fontsize=12, ha='center')
            pdf.savefig(fig_cover)
            plt.close(fig_cover)

            # Summary page
            fig_summary = plt.figure(figsize=(8, 6))
            plt.axis('off')
            summary_text = f"""Executive Summary

Total Sales: ₹{df['Monthly_Sales_INR'].sum():,.0f}
Average Health Score: {df['MSME_Health_Score'].mean():.1f}%
Average Risk Score: {df['Financial_Risk_Score'].mean():.2f}

6-Month Forecast: ₹{forecast_results['6_month']['forecast']:,.0f}
12-Month Forecast: ₹{forecast_results['12_month']['forecast']:,.0f}

Top Product: {df.nlargest(1, 'Monthly_Sales_INR')['SKU_Name'].values[0]}
"""
            plt.text(0.1, 0.5, summary_text, fontsize=12, family='monospace')
            pdf.savefig(fig_summary)
            plt.close(fig_summary)

            # Add dashboard charts
            if dashboard_figs:
                for fig in dashboard_figs:
                    if fig is not None:
                        pdf.savefig(fig)

        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
            return None, "PDF generation failed"

        return pdf_path, None

    except Exception as e:
        return None, f"PDF Error: {str(e)}"

# ==================== MOCK DATA ====================


# ==================== HELPER FUNCTIONS ====================

def _fetch_msme_data(msme_number):
    db = SessionLocal()
    try:
        record = db.query(UdyamMaster).filter(
            UdyamMaster.msme_number == msme_number
        ).first()

        if record:
            return (
                record.enterprise_name,
                record.organization_type,
                record.major_activity,
                record.enterprise_type,
                record.state,
                record.city,
                "✅ MSME Data Fetched Successfully"
            )
        else:
            return "", "", "", "", "", "", "❌ MSME Data Not Found"
    finally:
        db.close()


# ==================== GRADIO UI ====================

business_types = ["Choose Business Type", "FMCG", "Supermarket", "Clothing", "Electronics", "Manufacturing", "Services"]
roles = ["Business Owner", "Co-Founder", "Manager", "Analyst", "Store Manager"]

bg_path = os.path.abspath("C:\\Users\\Admin\\Desktop\\Datanetra\\Datanetra\\retail_bg.png")

# with gr.Blocks(title="MSME Intelligent Agent", theme=gr.themes.Soft()) as demo:
with gr.Blocks(
    
    title="MSME Intelligent Agent",
    theme=gr.themes.Base(),

    css="""
   
.gradio-container {
    background-image: url("https://i.ibb.co/Kjhg9DJq/retail-bg.png") !important;
    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
}

.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    background: rgba(0,0,0,0.55);
    z-index: 0;
}

    /* Full background */  

/* Animation */
@keyframes gradientMove {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

@keyframes slowZoom {
    from { background-size: 100%; }
    to { background-size: 110%; }
}


/* Proper center container */
.center-wrapper {
    
    display: flex;
    justify-content: center;
    align-items: stretch;   /* 👈 important */
    gap: 30px;
    width: 100%;
    min-height:80vh;
   
}

.login-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(15px);
    border-radius: 18px;
    padding: 30px;
    max-width: 380px;
    width: 100%;

    border: 1px solid rgba(255,255,255,0.3);
    box-shadow: 0 0 25px rgba(255,255,255,0.15);


    position: relative;
}
@keyframes orangeGlow {
    0% {
        box-shadow:
            0 0 8px rgba(255,107,61,0.6),
            0 0 25px rgba(255,107,61,0.45),
            0 0 60px rgba(255,107,61,0.25);
    }
    50% {
        box-shadow:
            0 0 12px rgba(255,140,66,0.9),
            0 0 35px rgba(255,107,61,0.7),
            0 0 90px rgba(255,140,66,0.4);
    }
    100% {
        box-shadow:
            0 0 8px rgba(255,107,61,0.6),
            0 0 25px rgba(255,107,61,0.45),
            0 0 60px rgba(255,107,61,0.25);
    }
}

.login-card {
    animation: none;
}
.login-card::before {
    content: "";
    position: absolute;
    inset: 0;
    padding: 1.5px; /* border thickness */
    border-radius: 18px;

    background: white;

    -webkit-mask:
        linear-gradient(#000 0 0) content-box,
        linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;

    pointer-events: none;
}

.login-card input:focus {
    border: 1px solid #ff6b3d !important;
    box-shadow: 0 0 20px rgba(255,255,255,0.25);
}

/* Override primary button properly */
.login-card button {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    transition: all 0.25s ease-in-out;
}

/* Hover effect */
/* Hover */
.login-card button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    box-shadow: 0 0 20px rgba(255,107,61,0.6);
    transform: translateY(-2px) scale(1.03);
}

/* Make buttons full width inside card */
.login-card button:active {
    transform: scale(0.97);
}


    /* Title */
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #ff6b3d;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Primary Button */
   
    .gr-button-primary {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600;
    color: white !important;
    transition: all 0.25s ease-in-out !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
       box-shadow: 0 0 20px rgba(37, 99, 235, 0.5) !important;
    transform: translateY(-2px) scale(1.02);
}
.login-card .gr-textbox,
.login-card .gr-dropdown {
    margin-bottom: 12px;
}
.login-card button:not(.primary) {
    background: transparent !important;
    border: 1px solid #444 !important;
    color: #ccc !important;
}

.login-card button:not(.primary):hover {
     border-color: #3b82f6 !important;
    color: #3b82f6 !important;
}
/* Slide Animation */
.step-slide {
    animation: slideIn 0.4s ease-in-out;
}

/* Keyframes */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateX(70px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}
/* Progress Wrapper */
.progress-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 30px;
    margin-bottom: 20px;
}

/* Circle */
.progress-step {
    width: 35px;
    height: 35px;
    border-radius: 50%;
    background: #222;
    color: #aaa;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    transition: all 0.3s ease;
}

/* Active Step */
.progress-step.active {
    background: linear-gradient(135deg, #ff6b3d, #ff8c42);
    color: white;
    box-shadow: 0 0 15px rgba(255,107,61,0.6);
}

/* Connecting Line */
.progress-line {
    width: 60px;
    height: 3px;
    background: #333;
}
/.verification-loader {
    display: flex;
    align-items: center;
    gap: 12px;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #3b82f6;
    color: #e2e8f0;
    font-weight: 600;
    box-shadow: 0 0 25px rgba(59,130,246,0.4);
    margin-bottom: 15px;
}

.spinner {
    width: 18px;
    height: 18px;
    border: 3px solid rgba(255,255,255,0.2);
    border-top: 3px solid #3b82f6;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.verification-success {
    background: linear-gradient(135deg, #065f46, #064e3b);
    border: 1px solid #22c55e;
    color: #d1fae5;
    padding: 14px;
    border-radius: 12px;
    font-weight: 600;
    text-align: center;
    box-shadow: 0 0 25px rgba(34,197,94,0.5);
}
.login-card input,
.login-card .gr-dropdown {
    transition: all 0.3s ease;
}

.login-card input:focus,
.login-card .gr-dropdown:focus-within {
    transform: scale(1.02);
}


@keyframes progressMove {
    0% { width: 0; }
    50% { width: 70%; }
    100% { width: 100%; }
}
/* KPI Cards */
.dashboard-card {
    background: linear-gradient(145deg,#111,#0a0a0a);
    padding: 25px;
    border-radius: 16px;
    border: 1px solid rgba(255,107,61,0.2);
    text-align: center;
    box-shadow: 0 0 20px rgba(255,107,61,0.15);
    transition: all 0.3s ease;
}

.dashboard-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 0 40px rgba(255,107,61,0.4);
}

/* Chart Cards */
.chart-card {
    background: linear-gradient(145deg,#0f0f0f,#050505);
    padding: 15px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 0 25px rgba(0,0,0,0.6);
    transition: all 0.3s ease;
}

.chart-card:hover {
    transform: scale(1.02);
    box-shadow: 0 0 35px rgba(255,107,61,0.25);
}
.card-dim {
    position: relative;
    opacity: 0.35;
    pointer-events: none;
    transition: all 0.5s ease;
}

.card-dim::before {
    content: "";
    position: absolute;
    inset: 0;
    background: rgba(0,0,0,0.55);
    border-radius: 18px;
}

.success-highlight {
    position: relative;
    margin-top: 20px;
}


@keyframes successPop {
    from { transform: scale(0.9); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
}
.fade-in {
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.first-login-card {
    max-width: 360px;
    height: auto !important;
    align-self: center !important;
    padding: 28px 30px;
    flex: 0 0 auto;
}
.second-login-card {
    max-width: 360px;
    height: auto !important;
    align-self: center !important;
    padding: 28px 30px;
    flex: 0 0 auto;
}
.login-card:hover {
    transform: translateY(-4px);
}


    .divider {
        text-align: center;
        margin: 20px 0;
        color: #64748b;
        font-size: 12px;
    }
    

/* HEADER */
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 40px;
}

.logo-section {
    display: flex;
    align-items: center;
    gap: 10px;
}

.logo-img {
    height: 40px;
}

.logo-text {
    font-size: 22px;
    font-weight: bold;
    color: white;
}

/* HERO SECTION */
.hero-section {
    text-align: center;
    padding: 40px;
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    border-radius: 16px;
    margin: 20px 40px;
}

.hero-title {
    font-size: 32px;
    color: white;
    font-weight: bold;
}

.hero-sub-tagline {
    font-size: 18px;
    color: #cbd5e1;
}

.hero-description {
    font-size: 14px;
    color: #e2e8f0;
}

/* HOW IT WORKS */
.section {
    margin: 40px;
}

.section-title {
    text-align: center;
    font-size: 26px;
    color: white;
    margin-bottom: 20px;
}



/* CAPABILITIES */
.capabilities-section {
    text-align: center;
    margin-top: 50px;
}

.capabilities-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    padding: 30px;
}

.capability-card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 16px;
}

/* FOOTER */
.footer-section {
    display: flex;
    justify-content: space-between;
    padding: 20px 40px;
    margin-top: 40px;
}

.section-container {
    # margin: 20px 40px;
    # padding: 25px 30px;

    # border-radius: 16px;

    # /* ONLY BORDER */
    # border: 1.5px solid rgba(59,130,246,0.7);

    # /* REMOVE BACKGROUND */
    # background: transparent !important;

    # /* subtle glow */
    # box-shadow: 0 0 12px rgba(59,130,246,0.25);

    # backdrop-filter: none !important;
    # -webkit-backdrop-filter: none !important;

    # transition: all 0.3s ease;
     margin: 20px 40px;
    padding: 25px 30px;

    border-radius: 16px;
    border: 1.5px solid rgba(59,130,246,0.8);

    background: transparent !important;

    box-shadow: none;
}

.section-container:hover {
    border-color: rgba(59,130,246,1);
    box-shadow: 0 0 18px rgba(59,130,246,0.4);
}

.llm-card {
    background: linear-gradient(145deg, #0f172a, #020617);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid rgba(59,130,246,0.4);
    box-shadow: 0 0 25px rgba(59,130,246,0.2);
    animation: fadeIn 0.5s ease;
}

.llm-title {
    font-size: 22px;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 15px;
}

.llm-section {
    margin-top: 18px;
}

.llm-section-title {
    font-size: 16px;
    font-weight: 600;
    color: #facc15;
    margin-bottom: 8px;
}

.llm-bullet {
    margin-left: 15px;
    color: #e2e8f0;
    margin-bottom: 6px;
}

.llm-footer {
    margin-top: 15px;
    color: #22c55e;
    font-weight: 600;
}


"""
) as demo:
 
    step_state = gr.State(-1)
    user_data_state = gr.State({})

     # Landing Page (Step 0)
    with gr.Column(visible=True) as landing_col:
        gr.HTML("""
                <div class="header-container">
                    <div class="logo-section">
                        <img src="https://i.postimg.cc/qRNQYbZJ/Data-Netra-Logo.jpg"
                            class="logo-img"
                            alt="DataNetra.ai Logo">
                        <div class="logo-text">DataNetra.ai</div>
                    </div>

                    <div class="nav-menu">
                    </div>
                </div>

                <div class="hero-section section-container">
                    <h1 class="hero-title">AI-Powered Retail Intelligence</h1>
                    <h2 class="hero-sub-tagline">Data with Vision. Decisions with Confidence.</h2>
                    <p class="hero-description">Turn retail data into predictive insights, smarter decisions, and measurable growth.</p>
                </div>
                """)
         # How DataNetra Works Section
        with gr.Column(elem_classes="section section-container", elem_id="how-datanetra-works-section"):
            gr.Markdown("## Get Started With DataNetra ", elem_classes="section-title")
            gr.Markdown("---")
            with gr.Row():
                # LEFT SIDE – Steps Description
                with gr.Column(scale=1, elem_classes="steps-description-column"):
                    gr.Markdown("### 📥 Step 1: Upload Your Data")
                    gr.Markdown("Easily upload Excel/CSV files (sales, product, operational data) for comprehensive analysis.")

                    gr.Markdown("### 🤖 Step 2: AI-Powered Analysis")
                    gr.Markdown("Our AI processes your data, identifying patterns, forecasting trends, and uncovering hidden insights.")

                    gr.Markdown("### 📊 Step 3: Actionable Dashboards & Recommendations")
                    gr.Markdown("Access interactive dashboards, KPI charts, and personalized recommendations to optimize inventory, boost sales, and drive confident decisions.")

                # RIGHT SIDE – Login/Signup Box
                with gr.Column(scale=1, elem_classes="login-signup-card section-container"):
                    gr.Markdown("**Already Registered**", elem_classes="login-signup-title")
                    quick_login_mobile = gr.Textbox(label="Enter Mobile Number", placeholder="+91")
                    quick_login_btn = gr.Button("Login", variant="primary", size="lg")
                    
                    landing_login_status = gr.Markdown("")   # ✅ ADD THIS LINE

                    gr.Markdown("**First Time User**", elem_classes="login-signup-title")
                    gr.Markdown("**Signup to unlock smart AI Insights**", elem_classes="login-signup-instruction")
                    quick_signup_btn = gr.Button("Sign Up Now", variant="primary", size="lg")

        # Platform Capabilities Section
        gr.HTML("""
        <div class="capabilities-section section-container">
            <h2 class="section-title">Platform Capabilities</h2>
            <p class="capabilities-tagline">Everything you need to make smarter business decisions</p>
            <div class="capabilities-grid">
                <div class="capability-card">
                    <div class="capability-icon">🎯</div>
                    <h3 class="capability-title">Smart Scoring Engine</h3>
                    <p class="capability-description">Automated analysis for accurate performance scoring and health scores</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">📊</div>
                    <h3 class="capability-title">Business Health Dashboard</h3>
                    <p class="capability-description">Monitor key metrics and KPIs in one real-time dashboard</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">🔮</div>
                    <h3 class="capability-title">Predictive Insights</h3>
                    <p class="capability-description">AI-driven forecasts to anticipate future trends</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">🔗</div>
                    <h3 class="capability-title">Easy Integration</h3>
                    <p class="capability-description">Seamlessly connect with your existing retail and POS systems</p>
                </div>
            </div>
        </div>

        <div class="quick-links-section">
            <h3 class="quick-links-title">Quick Links</h3>
            <div class="quick-links-grid">
                <a href="#" class="quick-link-item">About Us</a>
                <a href="#" class="quick-link-item">Features</a>
                <a href="#" class="quick-link-item">Documentation</a>
                <a href="#" class="quick-link-item">Help Center</a>
            </div>
        </div>

        <div class="footer-section">
            <div class="footer-left">
                <span class="footer-lock-icon">🔒</span>
                <span class="footer-text">Data Secured & Protected</span>
            </div>
            <div class="footer-right">
                <span class="footer-company">© 2026 DataNetra.ai</span>
                <a href="https://www.linkedin.com/company/108412762/admin/dashboard/" target="_blank" class="linkedin-link">
                    <span class="linkedin-icon">🔗</span>
                    <span style="color: white; font-size: 12px;">LinkedIn</span>

                </a>
            </div>
        </div>
        """)

    

    # Landing Page
    
    with gr.Column(visible=False, elem_classes="step-slide") as step0_col:

        gr.Markdown("<div class='main-title'>🚀 MSME IntelliCore</div>")
        gr.Markdown("<div class='sub-title'>AI-Powered Business Intelligence for MSMEs</div>")

        with gr.Row(elem_classes="center-wrapper"):

            with gr.Column(scale=0, elem_classes="login-card first-login-card"):

                gr.Markdown("🔐 **Already Registered on This Platform**")

                login_mobile = gr.Textbox(
                    placeholder="Mobile Number",
                    show_label=False
                )

                login_btn = gr.Button(
                    "Login / Check Status",
                    variant="primary"
                )

                gr.Markdown("<div class='divider'>OR</div>")

                gr.Markdown("✨ **First-Time MSME User on This Platform**")

                register_btn = gr.Button(
                    "Register New Account"
                )

                login_status = gr.Markdown("")

    # Step 1: User Information
    
    with gr.Column(visible=False, elem_classes="step-slide") as step1_col:

        with gr.Row(elem_classes="center-wrapper"):
            with gr.Column(scale=0, elem_classes="login-card second-login-card"):

                gr.Markdown("## 👤 User Information")

                name_input = gr.Textbox(
                    label="Full Name*",
                    placeholder="Enter your full name"
                )

                mobile_input = gr.Textbox(
                    label="Mobile Number*",
                    placeholder="Enter your mobile number"
                )

                email_input = gr.Textbox(
                    label="Email",
                    placeholder="Enter your email address"
                )

                role_input = gr.Dropdown(
                    choices=["Select Role"] + roles,
                    value="Select Role",
                    label="Role*"
                )


                with gr.Row():
                    cancel1_btn = gr.Button("Cancel")
                    next1_btn = gr.Button("Next →", variant="primary")

                error1 = gr.Markdown()

    
    # Step 2: MSME Verification (2 Column Layout)
    with gr.Column(visible=False, elem_classes="step-slide") as step2_col:

        with gr.Row(elem_classes="center-wrapper"):

            # LEFT CARD (Input Section)
            with gr.Column(scale=1, elem_classes="login-card"):

                gr.Markdown("## 🏢 MSME Verification")

                msme_number_input = gr.Textbox(
                    label="MSME / Udyam Number*",
                    placeholder="UDYAM-UP-01-0000001"
                )

                otp_input = gr.Textbox(
                    label="OTP*",
                    placeholder="Enter OTP (1234 for demo)",
                    type="password"
                )

                fetch_btn = gr.Button(
                    "🔎 Fetch MSME Data",
                    variant="primary"
                )

                fetch_status = gr.Markdown()

                with gr.Row():
                    back2_btn = gr.Button("← Back")
                    next2_btn = gr.Button("Verify & Next →", variant="primary")

                error2 = gr.Markdown()


            # RIGHT CARD (Fetched Details Section)
            with gr.Column(scale=1, elem_classes="login-card"):

                gr.Markdown("## 📄 Fetched MSME Details")

                fetched_name = gr.Textbox(label="Enterprise Name", interactive=False)
                fetched_org = gr.Textbox(label="Organisation Type", interactive=False)
                fetched_activity = gr.Textbox(label="Major Activity", interactive=False)
                fetched_type = gr.Textbox(label="Enterprise Type", interactive=False)
                fetched_state = gr.Textbox(label="State", interactive=False)
                fetched_city = gr.Textbox(label="City", interactive=False)

    # Step 3: Certificate Review (2 Column Layout)
    with gr.Column(visible=False, elem_classes="step-slide") as step3_col:

        with gr.Row(elem_classes="center-wrapper"):

            # LEFT CARD — Confirmed MSME Details
            with gr.Column(scale=1, elem_classes="login-card"):

                gr.Markdown("## 📄 Confirm MSME Details")

                confirm_name = gr.Textbox(
                    label="Enterprise Name",
                    interactive=False
                )

                confirm_org = gr.Textbox(
                    label="Organisation Type",
                    interactive=False
                )

                confirm_activity = gr.Textbox(
                    label="Major Activity",
                    interactive=False
                )

                confirm_type = gr.Textbox(
                    label="Enterprise Type",
                    interactive=False
                )

                confirm_state = gr.Textbox(
                    label="State",
                    interactive=False
                )

                confirm_city = gr.Textbox(
                    label="City",
                    interactive=False
                )


            # RIGHT CARD — Certificate & Consent
            with gr.Column(scale=1, elem_classes="login-card"):

                gr.Markdown("## ✅ Verification & Certificate")

                consent1 = gr.Checkbox(
                    label="I confirm the above MSME details are correct"
                )

                consent2 = gr.Checkbox(
                    label="I consent to verify the MSME certificate"
                )

                certificate_upload = gr.File(
                    label="Upload MSME Certificate (PDF)",
                    file_types=[".pdf"]
                )

                verification_loading = gr.Markdown(visible=False)
                verification_success = gr.Markdown(visible=False)

                with gr.Row():
                    back3_btn = gr.Button("← Back")
                    next3_btn = gr.Button("Confirm & Proceed →", variant="primary")

                error3 = gr.Markdown()
    
    # Step 4: Business Profile (Premium Layout)
    with gr.Column(visible=False, elem_classes="step-slide") as step4_col:

        with gr.Row(elem_classes="center-wrapper"):

            # LEFT CARD — Business Details
            with gr.Column(scale=1, elem_classes="login-card profile-form-card ") as profile_form_card:

                gr.Markdown("## 🏢 Business Profile Setup")

                verification_status_display = gr.Markdown(visible=False)

                business_type_input = gr.Dropdown(
                    choices=business_types,
                    label="Business Type*",
                    interactive=False   # make it auto-filled & locked
)


                years_input = gr.Number(
                    label="Years in Operation*",
                    value=1,
                    minimum=0
                )

                revenue_input = gr.Dropdown(
                    label="Monthly Revenue Range*",
                    choices=[
                        "< 5 Lakh",
                        "5-10 Lakh",
                        "10-50 Lakh",
                        "50 Lakh - 1 Crore",
                        "> 1 Crore"
                    ]
                )

                error4 = gr.Markdown()


            # RIGHT CARD — Actions
            with gr.Column(scale=1, elem_classes="login-card action-card ") as action_card:

                gr.Markdown("## 🚀 Complete Profile")

                gr.Markdown(
                    "Submit your business profile to unlock AI-powered insights, dashboard analytics, and smart forecasting."
                )

                with gr.Row():
                    back4_btn = gr.Button("← Back")
                    next4_btn = gr.Button("Submit Profile", variant="primary")

                gr.Markdown("---")

                proceed_to_step5_btn = gr.Button(
                    "✨ Next: Upload Business Data →",
                    variant="primary",
                    visible=False
                )
                success_card = gr.Markdown(
    visible=False,
    elem_classes="login-card fade-in"
)

    
    # Step 5: Data Upload & AI Analysis (Premium Layout)
    with gr.Column(visible=False, elem_classes="step-slide") as step5_col:

        with gr.Row(elem_classes="center-wrapper"):

            # LEFT CARD — Upload Section
            with gr.Column(scale=1, elem_classes="login-card"):

                gr.Markdown("## 📂 Upload Business Data")

                gr.Markdown(
                    "Upload your sales dataset to unlock AI insights, forecasting, "
                    "risk analysis, and performance dashboard."
                )

                consent_check = gr.Checkbox(
                    label="I consent to data analysis*"
                )

                file_upload = gr.File(
                    label="Upload Excel File (.xlsx, .csv)*",
                    file_types=[".xlsx", ".csv"]
                )

                upload_message = gr.Markdown(
                    value="",
                    visible=False
                )

                error5 = gr.Markdown()


            # RIGHT CARD — Actions & Results
            with gr.Column(scale=1, elem_classes="login-card"):

                gr.Markdown("## 🚀 Run AI Analysis")

                gr.Markdown(
                    "Our ML engine will generate:\n"
                    "- 📊 Performance Dashboard\n"
                    "- 🔮 Sales Forecast\n"
                    "- 🧠 Health Score\n"
                    "- 📄 Downloadable PDF Report"
                )

                with gr.Row():
                    back5_btn = gr.Button("← Back")
                    cancel5_btn = gr.Button("❌ Cancel")
                    analyze_btn = gr.Button("🚀 Analyze Data", variant="primary")

                gr.Markdown("---")

                insights_output = gr.Markdown()

                pdf_output = gr.File(label="Download PDF Report")

                view_dashboard_btn = gr.Button(
                    "📊 View Dashboard",
                    visible=False,
                    variant="primary"
                )

    # Step 6: Dashboard
    

    # ==================== EVENT HANDLERS ====================

    

    def update_visibility(step):
        return [
            gr.update(visible=(step == -1)),  # landing_col
            gr.update(visible=(step == 0)),   # step0_col
            gr.update(visible=(step == 1)),   # step1_col
            gr.update(visible=(step == 2)),   # step2_col
            gr.update(visible=(step == 3)),   # step3_col
            gr.update(visible=(step == 4)),   # step4_col
            gr.update(visible=(step == 5)),   # step5_col
        ]



    def handle_login(mobile):

        if not mobile or mobile.strip() == "":
            return (
                "⚠️ Please enter mobile number",
                {},
                -1,
                *update_visibility(-1)
            )

        profile = get_user_profile(mobile)

        if profile:

            return (
                f"✅ Welcome back, {profile['full_name']}",
                profile,
                5,
                *update_visibility(5)
            )

        else:

            return (
                "❌ Sorry, user not found. Please register first.",
                {},
                -1,
                *update_visibility(-1)
            )


        # return ("❌ No account found. Please register.", {}, 0, *update_visibility(0))

    def handle_register():
        return ("", 1, *update_visibility(1), gr.update(visible=False)) # Added gr.update for proceed_to_step5_btn

    def validate_step1(name, mobile, email, role, current_data):
        if not name or not mobile or not role:
            return ("⚠️ Please fill all required fields", current_data, 1, *update_visibility(1), gr.update(visible=False)) # Added gr.update for proceed_to_step5_btn

        updated_data = {
            **current_data,
            'full_name': name,
            'mobile_number': mobile,
            'email': email,
            'role': role
        }
        return ("", updated_data, 2, *update_visibility(2), gr.update(visible=False)) # Added gr.update for proceed_to_step5_btn

    def verify_step2(msme_num, otp, current_data,
                ent_name, org, activity, ent_type,
                state, city, status):

    # stay function
        def stay(message):
            return (
                message,            # error2
                current_data,       # user_data_state
                2,                  # step_state
                *update_visibility(2),  # landing_col → step6_col (8 values)
                ent_name,           # confirm_name
                org,                # confirm_org
                activity,           # confirm_activity
                ent_type,           # confirm_type
                state,              # confirm_state
                city,               # confirm_city
                gr.update(visible=False)  # proceed_to_step5_btn
            )

        # validations
        if not msme_num or not otp:
            return stay("⚠️ Please fill MSME number and OTP")

        if otp != "1234":
            return stay("⚠️ Invalid OTP")

        if "Successfully" not in status:
            return stay("⚠️ Please fetch MSME data first")

        # success case
        updated_data = {
            **current_data,
            'msme_number': msme_num,
            'company_name': ent_name,
            'organisation_type': org,
            'major_activity': activity,
            'enterprise_type': ent_type,
            'state': state,
            'city': city
        }

        return (
            "",                    # error2 clear
            updated_data,          # user_data_state
            3,                     # step_state
            *update_visibility(3), # landing_col → step6_col
            ent_name,
            org,
            activity,
            ent_type,
            state,
            city,
            gr.update(visible=False)
        )

    

    def confirm_step3(current_data, c1, c2, cert):

        if not c1 or not c2:
            return ("⚠️ Please accept both consents", current_data, 3, *update_visibility(3), gr.update(value="", visible=False), gr.update(visible=False))

        if cert is None:
            return ("⚠️ Please upload certificate", current_data, 3, *update_visibility(3), gr.update(value="", visible=False), gr.update(visible=False))

        # 🔥 STEP 1: Extract text from PDF
        with pdfplumber.open(cert.name) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        # 🔥 STEP 2: Compare with DB fetched data
        mismatches = []

        if current_data.get("msme_number") not in text:
            mismatches.append("MSME Number")

        if current_data.get("company_name") not in text:
            mismatches.append("Enterprise Name")

        if current_data.get("organisation_type") not in text:
            mismatches.append("Organisation Type")

        if current_data.get("major_activity") not in text:
            mismatches.append("Major Activity")

        if current_data.get("enterprise_type") not in text:
            mismatches.append("Enterprise Type")

        if current_data.get("state") not in text:
            mismatches.append("State")

        if current_data.get("city") not in text:
            mismatches.append("City")

        # 🔥 STEP 3: Decision
        if mismatches:
            return (
            f"❌ Certificate mismatch detected in: {', '.join(mismatches)}",
            current_data,
            3,
            *update_visibility(3),
            gr.update(value="", visible=False),
            gr.update(visible=False)
      )


        # ✅ If everything matches
        updated_data = {**current_data, 'verification_status': 'APPROVED'}

        success_msg = f"""
    ## ✅ Certificate Verified Successfully

    All certificate details match with Udyam Master data.

    You may proceed to Business Profile.
    """
        return (
            gr.update(value="", visible=False),
            updated_data,
            4,
            *update_visibility(4),
            gr.update(value=success_msg, visible=True),
            gr.update(visible=False),
            gr.update(value=current_data.get("major_activity"))   # 👈 ADD THIS
        )

        
    def submit_profile(business_type, years, revenue, current_data):

        if business_type == "Choose Business Type" or not revenue:
            return (
                "⚠️ Please complete all required fields",
                current_data,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False)
            )

        updated_data = {
            **current_data,
            "business_type": business_type,
            "years_operation": years,
            "monthly_revenue_range": revenue,
            "consent_given": True
        }

        try:
            user_id = save_user_profile(updated_data)

            success_msg = f"""
    ### ✅ Profile Created Successfully!

    Your User ID: **{user_id}**

    You can now proceed to upload business data.
    """

            return (
                "",
                updated_data,
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=success_msg, visible=True)
            )

        except Exception as e:
            return (
                f"❌ Error saving profile: {str(e)}",
                current_data,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False)
            )

    # def open_dashboard():
    #     def run():
    #       run_dash()

    #     threading.Thread(target=run, daemon=True).start()
    #     webbrowser.open("http://127.0.0.1:8050")
    #     return 5, *update_visibility(5)

    import threading
    import webbrowser

    # Keep track of the server state
    DASH_STARTED = False

    # def open_dashboard():
    #     global DASH_STARTED
        
    #     if not DASH_STARTED:
    #         def run():
    #             # Assuming run_dash() starts your Dash/Flask server
    #             run_dash()

    #         thread = threading.Thread(target=run, daemon=True)
    #         thread.start()
    #         DASH_STARTED = True
    #         print("Dash server started.")
    #     else:
    #         print("Dash server is already running.")

    #     # Always open the browser or refresh the tab
    #     webbrowser.open("http://127.0.0.1:8050")
        
    #     # Return your Gradio component updates
    #     return 5, *update_visibility(5)
    def open_dashboard():
            global DASH_STARTED

            if not DASH_STARTED:
                def run():
                    run_dash()

                threading.Thread(target=run, daemon=True).start()
                DASH_STARTED = True

            return 5, *update_visibility(5)

    def analyze_data(user_data, consent, file):
            # Clear previous messages/outputs before analysis
            initial_output_updates = [
                gr.update(value="", visible=False), # insights_output
                gr.update(value=None, visible=False), # pdf_output
                gr.update(visible=False), # view_dashboard_btn
                gr.update(value="### 💰 Total Sales\n—"), # kpi1
                gr.update(value="### 📈 Total Profit\n—"), # kpi2
                gr.update(value="### 🧠 Health Score\n—"), # kpi3
                gr.update(value="### 🚀 Growth Score\n—"), # kpi4
                None, None, None, None, # charts
                gr.update(value="", visible=False) # upload_message
            ]

            if not consent:
                return (f"⚠️ Please provide consent to analyze data", gr.update(value="", visible=True), *initial_output_updates[1:])
            if file is None:
                return (f"⚠️ Please upload an Excel or CSV file", gr.update(value="", visible=True), *initial_output_updates[1:])

            try:
                # Read the file
                if file.name.endswith('.xlsx'):
                    df = pd.read_excel(file.name)
                elif file.name.endswith('.csv'):
                    df = pd.read_csv(file.name)
                else:
                    return (f"❌ Unsupported file format. Please upload .xlsx or .csv", gr.update(value="", visible=True), *initial_output_updates[1:])
                
                                # -----------------------------
                # SAVE RAW DATA TO SUPABASE (REPLACE OLD)
                # -----------------------------
                table_name = "demo_rawdata1"

                with engine.connect() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name};"))
                    conn.commit()

                df.to_sql(
                    table_name,
                    con=engine,
                    if_exists='replace',
                    index=False,
                    method='multi'
                )

                # ===============================
                # LLM CSV INSIGHTS (Gemma Model)
                # ===============================

                print("Generating LLM insights from CSV...")

                temp_csv = "uploaded_data.csv"

                df.to_csv(temp_csv, index=False)

                llm_insights = generate_llm_insights(temp_csv)

                print("LLM insights generated")


                print(f"Table '{table_name}' replaced successfully in Supabase.")


                # ---- FIX COLUMN NAMES (ADD THIS HERE) ----
                if "Gross_Sales" in df.columns:
                    df["Monthly_Sales_INR"] = df["Gross_Sales"]

                if "Product_Name" in df.columns:
                    df["SKU_Name"] = df["Product_Name"]

                if "Profit_Margin_%" in df.columns:
                    df["Avg_Margin_Percent"] = df["Profit_Margin_%"]

                if "Quantity_Sold" in df.columns:
                    df["Monthly_Demand_Units"] = df["Quantity_Sold"]

                if "Total_Cost" in df.columns:
                    df["Monthly_Operating_Cost_INR"] = df["Total_Cost"]

                if "Outstanding_Amount" in df.columns:
                    df["Outstanding_Loan_INR"] = df["Outstanding_Amount"]

                                
                # ---- ADD DEFAULT COLUMNS IF MISSING ----
                if "Vendor_Delivery_Reliability" not in df.columns:
                    df["Vendor_Delivery_Reliability"] = 0.8   # default good vendor score

                if "Inventory_Turnover" not in df.columns:
                    df["Inventory_Turnover"] = 5   # default turnover value
# -----------------------------------------

                # Validate required columns
                required_cols = ['Monthly_Sales_INR', 'SKU_Name']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    return (f"❌ Missing required columns: {', '.join(missing_cols)}", gr.update(value="", visible=True), *initial_output_updates[1:])

                print(f"Processing file with {len(df)} rows and {len(df.columns)} columns")

                # Generate insights
                # insights, error_msg, forecast_data = generate_insights(user_data, df)
                # if error_msg:
                    # return (f"❌ {error_msg}", gr.update(value="", visible=True), *initial_output_updates[1:])

                print("Insights generated successfully")

                # Generate dashboard data
                kpi1, kpi2, kpi3, kpi4, fig1, fig2, fig3, fig4, dash_error = generate_dashboard_data(user_data, df)

                if dash_error:
                    print(f"Dashboard error: {dash_error}")
                else:
                    print("Dashboard generated successfully")

                # Generate PDF report
                pdf_path, pdf_error = generate_pdf_report(user_data, df, [fig1, fig2, fig3, fig4])

                if pdf_error:
                    print(f"PDF generation warning: {pdf_error}")
                else:
                    print(f"PDF generated at: {pdf_path}")

                # return (
                #     insights if insights else "✅ Analysis completed successfully",
                #     pdf_path,
                #     gr.update(visible=True),
                #     gr.update(value="", visible=False) # Clear upload_message
                # )
                final_insights = llm_insights

                return (
                    final_insights,
                    pdf_path,
                    gr.update(visible=True),
                    gr.update(value="", visible=False)
                )


            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"Analysis error:\n{error_trace}")
                return (
                    f"❌ Analysis failed: {str(e)}\n\nPlease check your file format and try again.",
                    gr.update(value="", visible=True),
                    *initial_output_updates[1:]
                )

        # Helper function to display upload message
    def handle_file_upload_change(user_data, file):
        if file is not None:
            user_name = user_data.get('full_name', 'User')
            msg = f"Thank you, {user_name}, for uploading the dataset file. Click on 'Analyze Data' to view AI Insights, Dashboard, and PDF Report."
            return gr.update(value=msg, visible=True), gr.update(value="", visible=False) # Show upload message, clear error5
        else:
            return gr.update(value="", visible=False), gr.update(value="", visible=False) # Clear upload message and error5

        # Wire up events
        
    login_btn.click(handle_login, [login_mobile], [login_status, user_data_state, step_state,landing_col, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col])
    register_btn.click(handle_register, [], [login_status, step_state, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, proceed_to_step5_btn]) # Added proceed_to_step5_btn here

    cancel1_btn.click(lambda: (-1, *update_visibility(-1), gr.update(visible=False)), [], [step_state,landing_col, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, proceed_to_step5_btn])
    next1_btn.click(validate_step1, [name_input, mobile_input, email_input, role_input, user_data_state], [error1, user_data_state, step_state,landing_col, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, proceed_to_step5_btn])

    fetch_btn.click(_fetch_msme_data, [msme_number_input], [fetched_name, fetched_org, fetched_activity, fetched_type, fetched_state, fetched_city, fetch_status])
    back2_btn.click(lambda: (1, *update_visibility(1), gr.update(visible=False)), [], [step_state,landing_col, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, proceed_to_step5_btn])
    next2_btn.click(verify_step2, [msme_number_input, otp_input, user_data_state, fetched_name, fetched_org, fetched_activity, fetched_type, fetched_state, fetched_city, fetch_status],
                [error2, user_data_state, step_state,landing_col, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, confirm_name, confirm_org, confirm_activity, confirm_type, confirm_state, confirm_city, proceed_to_step5_btn])

    back3_btn.click(lambda: (2, *update_visibility(2), gr.update(visible=False)), [], [step_state,landing_col, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, proceed_to_step5_btn])
    next3_btn.click(
    confirm_step3,
    [user_data_state, consent1, consent2, certificate_upload],
    [error3, user_data_state, step_state,
     landing_col, step0_col, step1_col, step2_col,
     step3_col, step4_col, step5_col,
     verification_status_display,
     proceed_to_step5_btn,
     business_type_input])


    back4_btn.click(lambda: (3, *update_visibility(3), gr.update(visible=False), gr.update(value="", visible=False), gr.update(visible=True)), [], [step_state,landing_col, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, proceed_to_step5_btn, error4, next4_btn])

    next4_btn.click(
    submit_profile,
        [business_type_input, years_input, revenue_input, user_data_state],
        [
            error4,
            user_data_state,
            proceed_to_step5_btn,
            next4_btn,
            profile_form_card,   
            success_card         
        ]
    )

    proceed_to_step5_btn.click(
            lambda: (5, *update_visibility(5), gr.update(value="", visible=False), gr.update(visible=False)), # Clear error4 message and hide itself
            [],
            [step_state,landing_col, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, error4, proceed_to_step5_btn]
        )

    back5_btn.click(lambda: (4, *update_visibility(4)), [], [step_state,landing_col, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col])
    cancel5_btn.click(lambda: (-1, *update_visibility(-1)), [], [step_state,landing_col, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col]) # Cancel button for Step 5
    analyze_btn.click(
    analyze_data,
    [user_data_state, consent_check, file_upload],
    [insights_output, pdf_output, view_dashboard_btn, upload_message])
    file_upload.change(handle_file_upload_change, inputs=[user_data_state, file_upload], outputs=[upload_message, error5])
    view_dashboard_btn.click(
    open_dashboard,
    [],
    [
        step_state,
        landing_col,
        step0_col,
        step1_col,
        step2_col,
        step3_col,
        step4_col,
        step5_col
    ]).then(
    None,
    None,
    js="() => window.open('http://127.0.0.1:8050','_blank')"
    )


    quick_login_btn.click(
        handle_login,                      
        [quick_login_mobile],           
        [
            landing_login_status,                 
            user_data_state,              
            step_state,                   
            landing_col,
            step0_col,
            step1_col,
            step2_col,
            step3_col,
            step4_col,
            step5_col
        ]
    )


    quick_signup_btn.click(
        lambda: (1, *update_visibility(1)),
        [],
        [
            step_state,
            landing_col,
            step0_col,
            step1_col,
            step2_col,
            step3_col,
            step4_col,
            step5_col
        ]
    )


    if __name__ == "__main__":
        print("=" * 60)
        print("🚀 MSME Intelligent Agent - Complete & Working")
        print("=" * 60)
        print("✅ Submit Profile - FIXED")
        print("✅ MSME Certificate Approval - WORKING")
        print("✅ AI Insights - WORKING")
        print("✅ Dashboard with 4 Charts - WORKING")
        print("=" * 60)
        demo.launch()


# python -m venv venv
# venv\Scripts\activate
# pip install gradio pandas numpy matplotlib sqlalchemy psycopg2-binary scikit-learn prophet openpyxl
