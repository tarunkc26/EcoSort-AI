import streamlit as st
import sqlite3
import json
import re
import pandas as pd
from datetime import datetime
from google import genai
from google.genai import types

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide"
)

DB_NAME = "ecosort_history.db"
MODEL_NAME = "gemini-3.5-flash-lite"

CATEGORIES = [
    "Recyclable",
    "Reusable",
    "Biodegradable",
    "Hazardous",
    "Needs Review"
]


# ---------------- DATABASE ----------------

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                object_name TEXT,
                material TEXT,
                category TEXT,
                confidence TEXT,
                advice TEXT,
                safety TEXT,
                points INTEGER
            )
        """)


def save_scan(result):
    points = 10

    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            INSERT INTO scans (
                created_at, object_name, material,
                category, confidence, advice, safety, points
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result["object_name"],
            result["material"],
            result["category"],
            result["confidence"],
            json.dumps(result["advice"]),
            result["safety"],
            points
        ))


def get_history():
    with sqlite3.connect(DB_NAME) as conn:
        return pd.read_sql_query(
            "SELECT * FROM scans ORDER BY id DESC", conn
        )


init_db()


# ---------------- AI ANALYSIS ----------------

def analyze_image(api_key, image_bytes, mime_type):
    client = genai.Client(api_key=api_key)

    prompt = """
You are EcoSort AI, a careful waste identification assistant.

Analyze the provided image and return ONLY one valid JSON object.
Do not use Markdown fences.

Use exactly these keys:
{
  "object_name": "short likely object name",
  "material": "likely material",
  "category": "Recyclable, Reusable, Biodegradable, Hazardous, or Needs Review",
  "confidence": "High, Medium, or Low",
  "reason": "short explanation of identification and category",
  "advice": [
    "first practical disposal or reuse suggestion",
    "second practical suggestion",
    "third practical suggestion"
  ],
  "safety": "important safety warning, or No special warning identified"
}

Rules:
- Choose Needs Review if the image is unclear or the material cannot be
  identified reliably.
- Batteries, e-waste, chemicals, medicines, and sharps may require special
  handling; do not advise putting them in ordinary recycling.
- Do not claim that an item is recyclable everywhere. Local rules differ.
- Do not claim to know the exact material from appearance alone.
- Never invent a precise environmental impact or a measured confidence percentage.
- Give simple, useful advice.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            ),
            prompt
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json"
        )
    )

    raw = response.text
    if not raw:
        raise ValueError("The AI returned an empty response.")

    # Handle occasional JSON formatting differences.
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)

    category = str(result.get("category", "Needs Review")).strip()
    valid_categories = {item.lower(): item for item in CATEGORIES}

    result["category"] = valid_categories.get(
        category.lower(), "Needs Review"
    )

    result["object_name"] = str(
        result.get("object_name", "Unidentified item")
    )
    result["material"] = str(
        result.get("material", "Unknown")
    )
    result["confidence"] = str(
        result.get("confidence", "Low")
    )
    result["reason"] = str(
        result.get("reason", "Identification is uncertain.")
    )
    result["safety"] = str(
        result.get("safety", "Use caution and check local guidance.")
    )

    advice = result.get("advice", [])
    if not isinstance(advice, list):
        advice = [str(advice)]
    result["advice"] = [str(item) for item in advice[:3]]

    return result


# ---------------- SIDEBAR ----------------

st.sidebar.title("♻️ EcoSort AI")
st.sidebar.caption("Smart Recycling Assistant")
st.sidebar.markdown("---")

# Read the API key automatically from Streamlit Secrets.
# If no secret is found (for example on a laptop), show the input box.
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = st.sidebar.text_input(
        "Gemini API key",
        type="password",
        help="Your key is used for AI requests. Do not share it."
    )

st.sidebar.info(
    "Capture or upload a waste photo, analyze it with AI, "
    "and track your scans in the dashboard."
)

st.sidebar.markdown("---")
st.sidebar.caption("Prototype: AI estimates may be incorrect.")


# ---------------- HEADER ----------------

st.title("♻️ EcoSort AI")
st.subheader("AI-powered waste identification and recycling guidance")
st.write(
    "Capture an item, discover its likely material, and learn how "
    "to sort it responsibly."
)

# ---------------- NAVIGATION ----------------

page = st.radio(
    "Choose a section",
    ["Scan Waste", "Digital Twin Dashboard", "Scan History"],
    horizontal=True
)


# ---------------- SCAN PAGE ----------------

if page == "Scan Waste":
    st.header("1. Capture or upload a waste item")

    camera_tab, upload_tab = st.tabs(
        ["📷 Live Camera", "🖼️ Upload Image"]
    )

    with camera_tab:
        photo = st.camera_input(
            "Point your camera at one waste item",
            key="camera_photo"
        )

    with upload_tab:
        uploaded = st.file_uploader(
            "Choose a photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="uploaded_photo"
        )

    image_file = photo if photo is not None else uploaded

    if image_file is not None:
        image_bytes = image_file.getvalue()
        mime_type = image_file.type or "image/jpeg"

        st.image(
            image_bytes,
            caption="Image selected for analysis",
            width=350
        )

        if st.button("🔍 Analyze with AI", type="primary"):
            if not api_key.strip():
                st.error(
                    "Gemini API key not found. Add GEMINI_API_KEY "
                    "in Streamlit Secrets."
                )
            else:
                try:
                    with st.spinner(
                        "AI is analyzing the image. Please wait..."
                    ):
                        result = analyze_image(
                            api_key.strip(),
                            image_bytes,
                            mime_type
                        )
                        save_scan(result)

                    # Store the result so it remains visible on reruns.
                    st.session_state["last_result"] = result
                    st.success("Analysis completed and saved!")

                except Exception as error:
                    st.error(
                        "Analysis failed. Check your internet connection, "
                        "API key, API access, model availability, and quota."
                    )
                    st.code(str(error))

    result = st.session_state.get("last_result")

    if result:
        st.markdown("---")
        st.header("2. AI Analysis Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Object", result["object_name"])
        col2.metric("Likely Material", result["material"])
        col3.metric("Category", result["category"])

        st.write("**AI confidence estimate:**", result["confidence"])
        st.write("**Why this category?**", result["reason"])

        st.markdown("### ♻️ Recommended actions")
        for advice_item in result["advice"]:
            st.write(f"- {advice_item}")

        st.markdown("### ⚠️ Safety and local rules")
        st.warning(result["safety"])
        st.caption(
            "AI results are estimates, not certified material tests. "
            "Check your local waste collection rules before disposal."
        )

        st.success("You earned 10 Eco Points for this saved scan.")


# ---------------- DIGITAL TWIN DASHBOARD ----------------

elif page == "Digital Twin Dashboard":
    st.header("🌍 Waste Digital Twin Dashboard")
    st.write(
        "A simple digital record of scanned waste items. "
        "Statistics update as you analyze more items."
    )

    df = get_history()

    if df.empty:
        st.info(
            "No scans yet. Go to Scan Waste and analyze your first item "
            "to populate this dashboard."
        )
    else:
        total_scans = len(df)
        recyclable = int((df["category"] == "Recyclable").sum())
        reusable = int((df["category"] == "Reusable").sum())
        biodegradable = int(
            (df["category"] == "Biodegradable").sum()
        )
        hazardous = int((df["category"] == "Hazardous").sum())
        points = int(df["points"].sum())

        a, b, c = st.columns(3)
        a.metric("Items scanned", total_scans)
        b.metric("Recyclable items", recyclable)
        c.metric("Eco Points", points)

        d, e = st.columns(2)
        d.metric("Reusable items", reusable)
        e.metric("Hazardous items", hazardous)

        st.markdown("### Waste categories")
        counts = (
            df["category"]
            .value_counts()
            .reindex(CATEGORIES, fill_value=0)
        )
        chart_df = counts.rename_axis("Category").reset_index(
            name="Number of scans"
        )
        st.bar_chart(
            chart_df,
            x="Category",
            y="Number of scans"
        )

        st.markdown("### Latest scans")
        st.dataframe(
            df[[
                "created_at", "object_name", "material",
                "category", "points"
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "This dashboard is a simple digital representation of scan "
            "records, not a live sensor-based physical waste facility. "
            "No landfill diversion or carbon savings are claimed."
        )


# ---------------- SCAN HISTORY ----------------

elif page == "Scan History":
    st.header("📋 Scan History")

    df = get_history()

    if df.empty:
        st.info("No saved scans yet. Analyze a waste item first.")
    else:
        search_text = st.text_input(
            "Search by object, material, or category"
        )

        filtered = df.copy()
        if search_text.strip():
            mask = (
                filtered["object_name"].str.contains(
                    search_text, case=False, na=False
                )
                | filtered["material"].str.contains(
                    search_text, case=False, na=False
                )
                | filtered["category"].str.contains(
                    search_text, case=False, na=False
                )
            )
            filtered = filtered[mask]

        st.dataframe(
            filtered[[
                "id", "created_at", "object_name", "material",
                "category", "confidence", "points"
            ]],
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download history as CSV",
            data=csv_data,
            file_name="ecosort_scan_history.csv",
            mime="text/csv"
        )


# ---------------- FOOTER ----------------

st.markdown("---")
st.caption(
    "EcoSort AI | Student hackathon prototype | "
    "Verify disposal advice against local guidelines."
)
