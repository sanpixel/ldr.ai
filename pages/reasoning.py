import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Classification Reasoning", layout="wide")

# CSS to make metric text smaller
st.markdown("""
<style>
/* Target all metric containers */
[data-testid="metric-container"] {
    font-size: 0.8rem !important;
}
[data-testid="metric-container"] * {
    font-size: inherit !important;
}
/* Target metric values (the large numbers/text) */
[data-testid="metric-container"] > div > div:first-child {
    font-size: 1.4rem !important;
}
/* Target metric labels (the titles) */
[data-testid="metric-container"] > div > div:last-child {
    font-size: 0.8rem !important;
}
/* Alternative targeting */
.metric-container {
    font-size: 0.8rem !important;
}
.stMetric > div {
    font-size: 0.8rem !important;
}
.stMetric > div > div {
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Classification Reasoning Dashboard")
st.markdown("Review how the AI classifies legal descriptions and makes decisions")

# Load reasoning data
reasoning_file = "classification_reasoning.json"

if os.path.exists(reasoning_file):
    try:
        with open(reasoning_file, 'r', encoding='utf-8') as f:
            reasoning_data = [json.loads(line) for line in f if line.strip()]
        
        if reasoning_data:
            st.success(f"Loaded {len(reasoning_data)} classification records")
            
            # Recent Classifications Overview
            st.subheader("📊 Recent Classifications")
            
            # Create summary stats
            df = pd.DataFrame(reasoning_data)
            if not df.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Classifications", len(df))
                with col2:
                    if 'classification' in df.columns:
                        most_common = df['classification'].mode().iloc[0] if not df['classification'].mode().empty else "N/A"
                        # Format the classification text to be more readable
                        if most_common != "N/A":
                            formatted_common = most_common.replace('_', ' ').title()
                        else:
                            formatted_common = "N/A"
                        st.metric("Most Common Type", formatted_common)
                with col3:
                    if 'confidence' in df.columns:
                        # Calculate confidence percentage - handle both 'High' and 'high'
                        high_conf = len(df[df['confidence'].str.lower() == 'high'])
                        high_conf_pct = round((high_conf / len(df)) * 100, 1) if len(df) > 0 else 0
                        st.metric("High Confidence", f"{high_conf_pct}%")
                with col4:
                    if 'timestamp' in df.columns:
                        # Show how many classifications were made today
                        try:
                            today = datetime.now().date()
                            today_count = 0
                            for _, row in df.iterrows():
                                try:
                                    entry_date = datetime.fromisoformat(row['timestamp']).date()
                                    if entry_date == today:
                                        today_count += 1
                                except:
                                    continue
                            st.metric("Today's Classifications", today_count)
                        except:
                            # Fallback to average confidence if timestamp parsing fails
                            if 'confidence' in df.columns:
                                conf_counts = df['confidence'].value_counts()
                                avg_label = "Avg Confidence"
                                if 'high' in conf_counts and conf_counts['high'] > len(df) * 0.6:
                                    avg_value = "High"
                                elif 'low' in conf_counts and conf_counts['low'] > len(df) * 0.6:
                                    avg_value = "Low"
                                else:
                                    avg_value = "Medium"
                                st.metric(avg_label, avg_value)
                            else:
                                st.metric("Data Available", "✓")
            
            # Detailed View
            st.subheader("🔍 Detailed Classifications")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                classification_filter = st.selectbox(
                    "Filter by Classification",
                    ["All"] + list(df['classification'].unique()) if 'classification' in df.columns else ["All"]
                )
            with col2:
                confidence_filter = st.selectbox(
                    "Filter by Confidence",
                    ["All"] + list(df['confidence'].unique()) if 'confidence' in df.columns else ["All"]
                )
            
            # Apply filters
            filtered_data = reasoning_data.copy()
            if classification_filter != "All":
                filtered_data = [r for r in filtered_data if r.get('classification') == classification_filter]
            if confidence_filter != "All":
                filtered_data = [r for r in filtered_data if r.get('confidence') == confidence_filter]
            
            # Display reasoning entries
            for i, entry in enumerate(reversed(filtered_data[-10:])):  # Show last 10
                with st.expander(f"Classification #{len(filtered_data)-i} - {entry.get('classification', 'Unknown')} ({entry.get('confidence', 'Unknown')} confidence)"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Input Text:**")
                        st.text_area("", entry.get('input_text', 'N/A')[:500] + "..." if len(entry.get('input_text', '')) > 500 else entry.get('input_text', 'N/A'), key=f"input_{i}", height=100)
                        
                        st.markdown("**AI Reasoning:**")
                        st.write(entry.get('reasoning', 'No reasoning provided'))
                        
                        if entry.get('evidence'):
                            st.markdown("**Evidence Found:**")
                            st.write(entry.get('evidence', 'No evidence provided'))
                        
                        if entry.get('alternatives'):
                            st.markdown("**Alternative Classifications:**")
                            alternatives_text = entry.get('alternatives', '')
                            if alternatives_text:
                                # Split by newlines since alternatives is stored as a joined string
                                alternatives_list = alternatives_text.split('\n')
                                for alt in alternatives_list:
                                    if alt.strip():  # Only show non-empty lines
                                        st.write(f"• {alt.strip()}")
                    
                    with col2:
                        st.markdown("**Classification Details:**")
                        st.json({
                            "Classification": entry.get('classification', 'Unknown'),
                            "Confidence": entry.get('confidence', 'Unknown'),
                            "Timestamp": entry.get('timestamp', 'Unknown')
                        })
            
            # Export functionality
            st.subheader("📤 Export Data")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Download JSON"):
                    st.download_button(
                        label="Download Classification Data",
                        data=json.dumps(reasoning_data, indent=2),
                        file_name=f"classification_reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            with col2:
                if st.button("Download CSV"):
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"classification_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No classification data found yet. Process some PDFs to see reasoning here!")
    
    except Exception as e:
        st.error(f"Error loading reasoning data: {str(e)}")
else:
    st.info("No reasoning file found yet. Classification reasoning will appear here after processing documents.")
    st.markdown("""
    **What you'll see here:**
    - 🎯 Classification decisions (explicit_bearings, abstract_bearings, external_ref)
    - 🤔 AI reasoning for each decision
    - 📊 Confidence levels and alternative choices
    - 📈 Summary statistics and trends
    - 🔍 Ability to filter and review specific cases
    """)

# Instructions
with st.expander("ℹ️ How to Use This Dashboard"):
    st.markdown("""
    1. **Process PDFs** in the main application to generate classification data
    2. **Review decisions** to understand how the AI classifies different document types
    3. **Identify patterns** in low-confidence classifications
    4. **Export data** to analyze trends or share with team
    5. **Use insights** to improve the classification prompt
    """)
