import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd
from utils.classification import (
    load_all_classification_data,
    get_filtered_classification_data, 
    clear_all_classification_data,
    get_classification_stats,
    test_database_connection
)

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

# Test database connection first
if not test_database_connection():
    st.error("❌ Cannot connect to classification database. Please check your connection.")
    st.stop()

# Load reasoning data from database
try:
    reasoning_data = load_all_classification_data()
    
    if reasoning_data:
        st.success(f"📊 Loaded {len(reasoning_data)} classification records from database")
        
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
        
        # Apply filters using database function for better performance
        filtered_data = get_filtered_classification_data(
            classification_filter if classification_filter != "All" else None,
            confidence_filter if confidence_filter != "All" else None,
            limit=50  # Limit for performance
        )
        
        # Display reasoning entries
        for i, entry in enumerate(filtered_data[:10]):  # Show first 10 of filtered results
            with st.expander(f"Classification #{i+1} - {entry.get('classification', 'Unknown')} ({entry.get('confidence', 'Unknown')} confidence)"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Input Text:**")
                    input_text = entry.get('input_text', 'N/A')
                    display_text = input_text[:500] + "..." if len(input_text) > 500 else input_text
                    st.text_area("", display_text, key=f"input_{i}", height=100)
                    
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
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Download JSON"):
                # Use filtered data for export
                export_data = filtered_data if classification_filter != "All" or confidence_filter != "All" else reasoning_data
                st.download_button(
                    label="Download Classification Data",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"classification_reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        with col2:
            if st.button("Download CSV"):
                export_df = pd.DataFrame(filtered_data) if classification_filter != "All" or confidence_filter != "All" else df
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"classification_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Admin section with clear database functionality
        with col3:
            with st.expander("🗑️ Admin Actions"):
                st.warning("⚠️ This will permanently delete ALL classification data!")
                clear_password = st.text_input("Enter password to clear:", type="password", key="clear_db_password")
                if clear_password == "warez":
                    if st.button("🗑️ CLEAR DATABASE", type="primary"):
                        if clear_all_classification_data():
                            st.success("✅ Database cleared successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to clear database")
                elif clear_password:
                    st.error("❌ Invalid password")
    else:
        st.info("📝 No classification data found in database. Process some PDFs to see reasoning here!")
        st.markdown("""
        **What you'll see here:**
        - 🎯 Classification decisions (explicit_bearings, abstract_bearings, external_ref)
        - 🤔 AI reasoning for each decision
        - 📊 Confidence levels and alternative choices
        - 📈 Summary statistics and trends
        - 🔍 Ability to filter and review specific cases
        """)

except Exception as e:
    st.error(f"❌ Error loading reasoning data: {str(e)}")
    st.info("Please check your database connection and try again.")

# Instructions
with st.expander("ℹ️ How to Use This Dashboard"):
    st.markdown("""
    1. **Process PDFs** in the main application to generate classification data
    2. **Review decisions** to understand how the AI classifies different document types
    3. **Identify patterns** in low-confidence classifications
    4. **Export data** to analyze trends or share with team
    5. **Use insights** to improve the classification prompt
    """)
