import pandas as pd
import time
from langchain_openai import ChatOpenAI


# ==============================
# OPENROUTER API KEY
# ==============================

OPENROUTER_API_KEY = "YOUR_API_KEY"


# ==============================
# INITIALIZE GEMMA MODEL
# ==============================

llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="google/gemma-3-4b-it:free",
    temperature=0
)


# ==============================
# MAIN FUNCTION
# ==============================

def generate_insights(csv_path):

    try:
        # ==============================
        # STEP 1: LOAD DATA
        # ==============================

        df = pd.read_csv(csv_path)

        # ==============================
        # STEP 2: CALCULATE EXACT METRICS
        # ==============================

        total_sales = df["Gross_Sales"].sum()

        total_profit = df["Profit_Amount"].sum()

        avg_margin = df["Profit_Margin_%"].mean()

        category_sales = (
            df.groupby("Product_Category")["Gross_Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        top_category = category_sales.index[0]
        top_category_sales = category_sales.iloc[0]

        store_sales = (
            df.groupby("Store_Location")["Gross_Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        top_store = store_sales.index[0]
        top_store_sales = store_sales.iloc[0]

        # ==============================
        # STEP 3: BUILD PROMPT WITH EXACT VALUES
        # ==============================

        prompt = f"""
You are a senior business intelligence analyst.

Analyze the following exact business metrics and generate professional dashboard insights.

DATA:

Total Sales: ₹{total_sales:,.0f}
Total Profit: ₹{total_profit:,.2f}
Average Profit Margin: {avg_margin:.2f}%

Top Product Category: {top_category}
Top Category Sales: ₹{top_category_sales:,.0f}

Top Store Location: {top_store}
Top Store Sales: ₹{top_store_sales:,.0f}

TASK:

Generate output in EXACT format below:

📊 Business Insights Summary
• Write 4 clear bullet insights using ONLY these values
• Do NOT change numbers
• Do NOT estimate

📌 Overall Business Summary
• Write 1–2 line executive summary

RULES:

• Use Indian Rupees symbol ₹
• Use professional business language
• Do NOT include technical explanations
"""

        # ==============================
        # STEP 4: CALL LLM WITH RETRY
        # ==============================

        max_retries = 2

        for attempt in range(max_retries):

            try:

                response = llm.invoke(prompt)

                insights = response.content

                formatted_output = f"""
{insights}

✅ Generated using Gemma LLM
"""

                return formatted_output

            except Exception as e:

                if "429" in str(e):

                    print("LLM busy, retrying...")
                    time.sleep(5)

                else:
                    return f"Error generating insights: {str(e)}"

        return """
⚠️ AI service is currently busy.
Please try again later.
"""

    except FileNotFoundError:

        return "Error: CSV file not found."

    except Exception as e:

        return f"Unexpected error: {str(e)}"
