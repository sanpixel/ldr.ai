import streamlit as st
import sys
import os
from io import BytesIO

# More reliable import approach for Cloud Run
try:
    from main import process_pdf, initialize_session_state
except ImportError:
    # Fallback for Cloud Run environment
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from main import process_pdf, initialize_session_state

st.set_page_config(page_title="Snapfinger Tract 2", layout="wide")

st.title("🧪 Test: Snapfinger Tract 2")
st.markdown("**File**: `combine_SNAPFINGER_TRACT-2_LD.pdf`")

# Initialize session state
initialize_session_state()

st.markdown("---")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Process Test PDF")
    st.write("Click the button below to process the second Snapfinger tract test PDF.")
    
    if st.button("🔄 Process Snapfinger Tract 2 PDF", use_container_width=True, type="primary"):
        try:
            # Check if file exists
            pdf_path = "combine_SNAPFINGER_TRACT-2_LD.pdf"
            if not os.path.exists(pdf_path):
                st.error(f"❌ Test PDF not found: {pdf_path}")
                st.info("Make sure the PDF file is in the root directory of the project.")
            else:
                with open(pdf_path, "rb") as test_pdf:
                    file_content = test_pdf.read()
                
                with st.spinner('Processing Snapfinger Tract 2 PDF...'):
                    # Create a BytesIO object to simulate uploaded file
                    pdf_file = BytesIO(file_content)
                    bearings = process_pdf(pdf_file)
                    
                    if bearings:
                        st.session_state.parsed_bearings = bearings
                        st.session_state.line_count = len(bearings)
                        
                        # Populate session state with extracted bearings
                        for i, bearing in enumerate(bearings):
                            st.session_state[f"cardinal_ns_{i}"] = bearing.get('cardinal_ns', "North")
                            st.session_state[f"degrees_{i}"] = bearing.get('degrees', 0)
                            st.session_state[f"minutes_{i}"] = bearing.get('minutes', 0)
                            st.session_state[f"seconds_{i}"] = bearing.get('seconds', 0)
                            st.session_state[f"cardinal_ew_{i}"] = bearing.get('cardinal_ew', "East")
                            st.session_state[f"distance_{i}"] = float(bearing.get('distance', 0.0))
                            st.session_state[f"monument_{i}"] = bearing.get('monument', '')
                        
                        st.success(f"✅ Successfully extracted and populated {len(bearings)} bearings from Snapfinger Tract 2!")
                        st.info("💡 Go to the main page to view and edit the extracted bearings.")
                    else:
                        st.warning("⚠️ No bearings found in this PDF.")
                        
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")

with col2:
    st.markdown("### File Info")
    pdf_path = "combine_SNAPFINGER_TRACT-2_LD.pdf"
    if os.path.exists(pdf_path):
        file_size = os.path.getsize(pdf_path)
        st.metric("File Size", f"{file_size:,} bytes")
        st.metric("Status", "✅ Found")
    else:
        st.metric("Status", "❌ Not Found")

# Show current session state info
if st.session_state.parsed_bearings:
    st.markdown("---")
    st.markdown("### Current Results")
    st.info(f"📊 Currently have {len(st.session_state.parsed_bearings)} bearings in session state")
    
    with st.expander("View Extracted Bearings Summary"):
        for i, bearing in enumerate(st.session_state.parsed_bearings):
            st.write(f"**Line {i+1}**: {bearing.get('original_text', 'No text')} - {bearing.get('distance', 0)} feet")
