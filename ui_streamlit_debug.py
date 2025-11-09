import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
import time

# -------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from hybrid_classifier_v3_clean_output import classify_single_item
except Exception as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="MRO Category Intelligence Dashboard", layout="wide")

# --- Title Row with Reset + Instructions ---
col1, col2, col3 = st.columns([5, 1, 1])
with col1:
    st.title("🫧 MRO Category Intelligence Dashboard")
with col2:
    if st.button("📘 Instructions"):
        st.session_state["show_instructions"] = not st.session_state.get("show_instructions", False)
with col3:
    if st.button("🔁 Reset Dashboard"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

# -------------------------------------------------------------------
# INSTRUCTIONS POPUP (COLLAPSIBLE)
# -------------------------------------------------------------------
if st.session_state.get("show_instructions", False):
    with st.expander("📘 How to Use the Dashboard (click to hide)", expanded=True):
        st.markdown("""
        ### 🧭 How to Use This Dashboard

        1️⃣ **Prepare your input Excel file**  
        - Must contain **only one column** named **`Item Description`** (exact spelling).  
        - Each row should represent one material, product, or spare part description.  

        2️⃣ **Upload your file above.**  
        - The system will automatically categorize the items into Level 1–3 taxonomy levels.  

        3️⃣ **Click ‘🚀 Run Auto-Categorization’**  
        - Watch progress in real time as the AI classifies your data.  

        4️⃣ **Review the Visualization**  
        - Bubble chart: Each bubble = Level 1 category  
        - Color = average confidence score  
        - Size = number of unique descriptions  

        5️⃣ **Download the Final Excel Results**  
        - Includes category mapping and confidence scores for every line item.  

        ---
        **⚙️ Input Template:** Download the official sample template below:
        """)

        template_path = BASE_DIR / "data" / "template_mro_input.xlsx"
        if template_path.exists():
            with open(template_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Excel Template",
                    data=f,
                    file_name="template_mro_input.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="template_dl",
                )
        else:
            st.warning("⚠️ Template file not found. Please ensure it exists under `data/template_mro_input.xlsx`.")
        st.info("Close this section once done reviewing instructions.")

# -------------------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload Excel file", type=["xlsx"], key="upload")
if not uploaded_file:
    st.info("Please upload an Excel file with an 'Item Description' column to begin.")
    st.stop()

df = pd.read_excel(uploaded_file)
if "Item Description" not in df.columns:
    st.error("❌ Missing 'Item Description' column. Please use the provided template.")
    st.stop()

st.success(f"✅ Loaded {len(df)} items for classification.")

# -------------------------------------------------------------------
# RUN BUTTON
# -------------------------------------------------------------------
run_button = st.button("🚀 Run Auto-Categorization", use_container_width=True, key="run_button")

progress = st.progress(0)
status = st.empty()
run_info_placeholder = st.empty()

# -------------------------------------------------------------------
# MAIN CLASSIFICATION LOOP
# -------------------------------------------------------------------
if run_button:
    start_time = time.time()
    run_info_placeholder.info("⏳ Running categorization... please wait until completion.")

    results = []
    n_total = len(df)
    batch_size = 20

    for i in range(0, n_total, batch_size):
        batch = df.iloc[i : i + batch_size]
        for _, row in batch.iterrows():
            desc = str(row.get("Item Description", "")).strip()
            results.append(classify_single_item(desc))

        progress.progress(min((i + batch_size) / n_total, 1.0))
        status.text(f"Processing items {i + 1} – {min(i + batch_size, n_total)} / {n_total}")
        time.sleep(0.2)

    # -------------------------------------------------------------------
    # FINAL OUTPUT
    # -------------------------------------------------------------------
    result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    run_info_placeholder.empty()
    status.empty()

    st.success(f"✅ Classification completed in {int(time.time() - start_time)} seconds!")

    output_path = Path("outputs") / "mro_classified_bubbles.xlsx"
    output_path.parent.mkdir(exist_ok=True)
    result_df.to_excel(output_path, index=False)

    # -------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------
    st.markdown("### 🌐 Category Confidence Landscape")

    summary = (
        result_df.groupby("Final_Level1", as_index=False)
        .agg(
            AvgConfidence=("Final_Score", "mean"),
            UniqueDescriptions=("Item Description", "nunique"),
            LineItems=("Final_Level1", "count"),
        )
    )
    summary["% of Total Items"] = (
        summary["LineItems"] / summary["LineItems"].sum() * 100
    ).round(2)
    summary = summary.sort_values("AvgConfidence", ascending=False)

    fig = px.scatter(
        summary,
        x="AvgConfidence",
        y="% of Total Items",
        size="UniqueDescriptions",
        color="AvgConfidence",
        text="Final_Level1",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        size_max=120,
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(line=dict(width=1.2, color="rgba(255,255,255,0.6)"), opacity=0.9),
    )

    fig.update_layout(
        title={
            "text": "🌐 Category Confidence Landscape — Bubble size = # Unique Descriptions, Color = Avg Confidence",
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": "white", "size": 16},
        },
        xaxis_title="Average Confidence →",
        yaxis_title="% of Total Line Items ↑",
        xaxis=dict(showgrid=False, tickfont=dict(color="white")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.15)", tickfont=dict(color="white")),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white"),
        showlegend=False,
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------
    # TABLE
    # -------------------------------------------------------------------
    st.markdown("### 📊 Category Summary Table (sorted by Avg Confidence)")

    styled_df = summary[
        ["Final_Level1", "UniqueDescriptions", "AvgConfidence", "% of Total Items"]
    ].rename(
        columns={
            "Final_Level1": "Level 1 Category",
            "UniqueDescriptions": "# Unique Descriptions",
            "AvgConfidence": "Avg Confidence Score",
            "% of Total Items": "% of Total Line Items",
        }
    )

    st.dataframe(
        styled_df.style.format(
            {"Avg Confidence Score": "{:.2f}", "% of Total Line Items": "{:.2f}"}
        ),
        use_container_width=True,
        height=400,
    )

    # -------------------------------------------------------------------
    # DOWNLOAD (NO RESET)
    # -------------------------------------------------------------------
    with open(output_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Full Results as Excel",
            data=f,
            file_name="mro_classified_bubbles.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="download_results",
            on_click=None
        )
