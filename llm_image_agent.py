

import base64
import os
import time
from openai import OpenAI

OPENROUTER_API_KEY = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=60
)

def encode_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


def generate_image_insights(image_path):

    try:

        if not os.path.exists(image_path):
            return "Error: Image not found."

        image_base64 = encode_image(image_path)

        prompt = """
Analyze this business dashboard chart.

STRICT OUTPUT FORMAT:

📊 Chart Insights Summary
• Provide 4 insights

📌 Overall Chart Summary
• Provide executive summary
"""

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

            error_str = str(e)

            # Credit exhausted / 402 error
            if "402" in error_str or "credit" in error_str.lower():
                return """📊 Chart Insights Summary
        • Chart explanation is temporarily unavailable.

        📌 Overall Chart Summary
        • AI insights service is currently unavailable. Please try again later.
        """

            # Rate limit
            elif "429" in error_str:
                return """📊 Chart Insights Summary
        • AI service is currently busy.

        📌 Overall Chart Summary
        • Please try again after a few moments.
        """

            # Any other error
            else:
                return """📊 Chart Insights Summary
        • Unable to generate chart insights.

        📌 Overall Chart Summary
        • Please try again later or contact support if the issue persists.
        """