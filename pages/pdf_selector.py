import streamlit as st
import sys
import os
import glob
from io import BytesIO

# More reliable import approach for Cloud Run
try:
    from main import process_pdf, initialize_session_state
except ImportError:
    # Fallback for Cloud Run environment
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from main import process_pdf, initialize_session_state

st.set_page_config(page_title="PDF Selector", layout="wide")

st.title("📄 PDF Selector")
st.markdown("Select any PDF file from the project directory to process")

# Initialize session state
initialize_session_state()

st.markdown("---")

# Get all PDF files in the root directory
pdf_files = glob.glob("*.pdf")

if not pdf_files:
    st.warning("⚠️ No PDF files found in the project root directory.")
    st.info("Place PDF files in the root directory to use this selector.")
else:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Available PDF Files")
        
        # Create a selectbox with all PDF files
        selected_pdf = st.selectbox(
            "Choose a PDF file to process:",
            options=pdf_files,
            index=0
        )
        
        if selected_pdf:
            # Display file information
            file_size = os.path.getsize(selected_pdf)
            st.write(f"**Selected**: `{selected_pdf}`")
            st.write(f"**Size**: {file_size:,} bytes")
            
            if st.button(f"🔄 Process {selected_pdf}", use_container_width=True, type="primary"):
                try:
                    with open(selected_pdf, "rb") as pdf_file:
                        file_content = pdf_file.read()
                    
                    with st.spinner(f'Processing {selected_pdf}...'):
                        # Create a BytesIO object to simulate uploaded file
                        pdf_buffer = BytesIO(file_content)
                        bearings = process_pdf(pdf_buffer)
                        
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
                            
                            st.success(f"✅ Successfully extracted and populated {len(bearings)} bearings from {selected_pdf}!")
                            st.info("💡 Go to the main page to view and edit the extracted bearings.")
                        else:
                            st.warning("⚠️ No bearings found in this PDF.")
                            
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
    
    with col2:
        st.markdown("### PDF Files Found")
        st.metric("Total PDFs", len(pdf_files))
        
        # Show all available PDFs with their sizes
        with st.expander("All PDF Files"):
            for pdf in pdf_files:
                size = os.path.getsize(pdf)
                st.write(f"• `{pdf}` ({size:,} bytes)")

# Show current session state info
if st.session_state.parsed_bearings:
    st.markdown("---")
    st.markdown("### Current Results")
    st.info(f"📊 Currently have {len(st.session_state.parsed_bearings)} bearings in session state")
    
    with st.expander("View Extracted Bearings Summary"):
        for i, bearing in enumerate(st.session_state.parsed_bearings):
            st.write(f"**Line {i+1}**: {bearing.get('original_text', 'No text')} - {bearing.get('distance', 0)} feet")
