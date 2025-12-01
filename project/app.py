import streamlit as st
from pathlib import Path
import base64
import json
from process_pdf import load_model, process_uploaded_pdf  # ✅ updated import

st.set_page_config(page_title="🧠 Automatic Extraction Project", page_icon="📄", layout="wide")

# ---- Header ----
st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: #3B82F6;'>📄 PDF Upload Portal</h1>
        <p style='font-size: 1.1rem; color: #6B7280;'>Easily upload and preview your research or document PDFs</p>
    </div>
""", unsafe_allow_html=True)

# ---- Load Model Once ----
@st.cache_resource
def load_models_once():
    return load_model()

tokenizer, model = load_models_once()

# ---- PDF Upload Section ----
st.markdown("---")
uploaded_file = st.file_uploader("📤 Upload your PDF file", type="pdf", help="Only PDF files are supported.")

# ---- Display Info ----
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    # Save file temporarily
    pdf_path = Path("temp") / uploaded_file.name
    pdf_path.parent.mkdir(exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show basic details
    st.markdown(f"**File name:** `{uploaded_file.name}`")
    st.markdown(f"**File size:** `{round(len(uploaded_file.getvalue()) / 1024, 2)} KB`")

    # Provide download option
    b64_pdf = base64.b64encode(uploaded_file.getvalue()).decode()
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{uploaded_file.name}" style="color: #10B981;">📥 Download PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Show the PDF inside the app
    st.markdown("---")
    st.markdown("### 📄 Preview")
    st.components.v1.html(f"""
        <iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>
    """, height=650)

    # ---- Extract Information ----
    st.markdown("---")
    st.markdown("### 🧠 Extracted Information")

    with st.spinner("Processing the PDF..."):
        result = process_uploaded_pdf(str(pdf_path), tokenizer, model)

    if "error" in result:
        st.error(f"❌ {result['error']}")
    else:
        # Show core sections
        st.subheader("📚 Sections")
        for key, value in result.get("sections", {}).items():
            st.markdown(f"**{key.capitalize()}**")
            st.text_area("", value, height=150)

        # Show named entities
        st.subheader("🔍 Named Entities")
        for entity, label in result.get("named_entities", []):
            st.markdown(f"- `{entity}` → **{label}**")

        # Show core terms
        st.subheader("🧾 Core Terms")
        st.markdown(", ".join(result.get("core_terms", [])))

        # Show LayoutLM tokens
        if result.get("tokens_with_labels"):
            st.subheader("📌 Token Labels (LayoutLM)")
            st.write(result["tokens_with_labels"])

        # Allow download of extracted JSON
        st.download_button("📥 Download JSON", json.dumps(result, indent=2), file_name="extracted_output.json", mime="application/json")

else:
    st.info("⬆️ Please upload a PDF file to begin.")

# ---- Footer ----
st.markdown("""
    <hr style="border: 1px solid #eaeaea;"/>
    <div style='text-align: center; color: gray; font-size: 0.9rem;'>
        © 2025 Automatic Extraction Project
    </div>
""", unsafe_allow_html=True)
