"""
Gold dataset collection UI components for Streamlit
Provides interface for manual verification and correction of outputs
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from utils.gold_dataset import (
    GoldDatasetManager, save_gold_entry, get_gold_entry_by_text,
    collect_real_descriptions_for_gold, create_gold_from_extraction
)
from utils.schema import SchemaOutput, LineData, LineType, BucketClassification
from utils.cached_rules_loader import get_current_processor
import json


def display_gold_dataset_dashboard():
    """
    Display gold dataset management dashboard
    """
    st.header("🏆 Gold Dataset Management")
    
    manager = GoldDatasetManager()
    stats = manager.get_cached_stats()
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Entries", stats.get('total_entries', 0))
    
    with col2:
        bucket_dist = stats.get('bucket_distribution', {})
        explicit_count = bucket_dist.get('explicit_bearings', 0)
        st.metric("Explicit Bearings", explicit_count)
    
    with col3:
        abstract_count = bucket_dist.get('abstract_bearings', 0)
        st.metric("Abstract Bearings", abstract_count)
    
    with col4:
        no_bearings_count = bucket_dist.get('no_bearings', 0)
        st.metric("No Bearings", no_bearings_count)
    
    # Recommendations
    recommendations = manager.get_recommended_additions()
    if recommendations:
        st.warning("**Recommendations for improving gold dataset:**")
        for rec in recommendations:
            st.write(f"• {rec}")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Collect New Candidates", use_container_width=True):
            st.session_state.show_collection_interface = True
    
    with col2:
        if st.button("✏️ Manual Entry", use_container_width=True):
            st.session_state.show_manual_entry = True
    
    with col3:
        if st.button("📊 View All Entries", use_container_width=True):
            st.session_state.show_all_entries = True
    
    # Show interfaces based on button clicks
    if st.session_state.get('show_collection_interface'):
        display_candidate_collection_interface()
    
    if st.session_state.get('show_manual_entry'):
        display_manual_entry_interface()
    
    if st.session_state.get('show_all_entries'):
        display_all_entries_interface()


def display_candidate_collection_interface():
    """
    Interface for collecting and reviewing candidate descriptions
    """
    st.subheader("📥 Collect Candidate Descriptions")
    
    # Collection parameters
    col1, col2 = st.columns(2)
    
    with col1:
        limit = st.number_input("Number of candidates to collect", 
                               min_value=5, max_value=100, value=20)
    
    with col2:
        if st.button("🔍 Collect Candidates"):
            with st.spinner("Collecting candidate descriptions..."):
                candidates = collect_real_descriptions_for_gold(limit)
                st.session_state.candidates = candidates
                st.success(f"Collected {len(candidates)} candidates")
    
    # Display candidates for review
    if 'candidates' in st.session_state and st.session_state.candidates:
        st.write(f"**Found {len(st.session_state.candidates)} candidates:**")
        
        for i, candidate in enumerate(st.session_state.candidates):
            with st.expander(f"Candidate {i+1}: {candidate.get('filename', 'Unknown')} - {candidate.get('classification', 'Unknown')}"):
                
                # Display candidate info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Text:**")
                    st.text_area("", candidate['text'], height=100, key=f"text_{i}", disabled=True)
                
                with col2:
                    st.write("**Classification Info:**")
                    st.write(f"Classification: {candidate.get('classification', 'Unknown')}")
                    st.write(f"Confidence: {candidate.get('confidence', 'Unknown')}")
                    st.write(f"Source: {candidate.get('filename', 'Unknown')}")
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"✅ Add to Gold", key=f"add_{i}"):
                        if add_candidate_to_gold(candidate):
                            st.success("Added to gold dataset!")
                            # Remove from candidates
                            st.session_state.candidates.pop(i)
                            st.rerun()
                
                with col2:
                    if st.button(f"✏️ Edit & Add", key=f"edit_{i}"):
                        st.session_state.editing_candidate = candidate
                        st.session_state.editing_index = i
                        st.rerun()
                
                with col3:
                    if st.button(f"❌ Skip", key=f"skip_{i}"):
                        st.session_state.candidates.pop(i)
                        st.rerun()
    
    # Handle editing
    if 'editing_candidate' in st.session_state:
        display_candidate_editor(st.session_state.editing_candidate, 
                                st.session_state.get('editing_index', 0))


def display_candidate_editor(candidate: Dict[str, Any], index: int):
    """
    Interface for editing a candidate before adding to gold dataset
    """
    st.subheader("✏️ Edit Candidate")
    
    # Run current extractor on the text
    text = candidate['text']
    
    with st.spinner("Running current extractor..."):
        processor = get_current_processor()
        extraction_results = processor.extract_all_patterns(text)
    
    # Display extraction results
    st.write("**Current Extractor Results:**")
    if extraction_results:
        st.json(extraction_results)
    else:
        st.info("No patterns extracted by current rules")
    
    # Manual correction interface
    st.write("**Manual Correction:**")
    
    # Bucket classification
    bucket_options = [bc.value for bc in BucketClassification]
    current_bucket = candidate.get('classification', 'no_bearings')
    if current_bucket not in bucket_options:
        current_bucket = 'no_bearings'
    
    corrected_bucket = st.selectbox(
        "Bucket Classification",
        bucket_options,
        index=bucket_options.index(current_bucket)
    )
    
    # Line editing
    st.write("**Lines:**")
    
    # Initialize lines in session state
    if f'lines_{index}' not in st.session_state:
        st.session_state[f'lines_{index}'] = []
    
    lines = st.session_state[f'lines_{index}']
    
    # Add line button
    if st.button("➕ Add Line"):
        new_line = {
            'type': LineType.COURSE.value,
            'idx': len(lines) + 1,
            'raw': '',
            'cardinal_ns': None,
            'degrees': None,
            'minutes': None,
            'seconds': None,
            'cardinal_ew': None,
            'distance': None,
            'monument': None,
            'reference': None
        }
        lines.append(new_line)
        st.session_state[f'lines_{index}'] = lines
        st.rerun()
    
    # Edit existing lines
    for i, line in enumerate(lines):
        with st.expander(f"Line {i+1}: {line.get('type', 'Unknown')}"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                line['type'] = st.selectbox(
                    "Type", 
                    [lt.value for lt in LineType],
                    index=[lt.value for lt in LineType].index(line.get('type', LineType.COURSE.value)),
                    key=f"type_{index}_{i}"
                )
                
                line['raw'] = st.text_input("Raw Text", line.get('raw', ''), key=f"raw_{index}_{i}")
            
            with col2:
                if line['type'] in [LineType.COURSE.value, LineType.CURVE.value]:
                    line['cardinal_ns'] = st.selectbox(
                        "Cardinal NS", 
                        [None, 'North', 'South'],
                        index=0 if line.get('cardinal_ns') is None else (['North', 'South'].index(line['cardinal_ns']) + 1),
                        key=f"ns_{index}_{i}"
                    )
                    
                    line['degrees'] = st.number_input("Degrees", value=line.get('degrees') or 0, key=f"deg_{index}_{i}")
                    line['minutes'] = st.number_input("Minutes", value=line.get('minutes') or 0, key=f"min_{index}_{i}")
                    line['seconds'] = st.number_input("Seconds", value=line.get('seconds') or 0.0, key=f"sec_{index}_{i}")
                    
                    line['cardinal_ew'] = st.selectbox(
                        "Cardinal EW", 
                        [None, 'East', 'West'],
                        index=0 if line.get('cardinal_ew') is None else (['East', 'West'].index(line['cardinal_ew']) + 1),
                        key=f"ew_{index}_{i}"
                    )
                    
                    line['distance'] = st.number_input("Distance", value=line.get('distance') or 0.0, key=f"dist_{index}_{i}")
                    line['monument'] = st.text_input("Monument", line.get('monument') or '', key=f"mon_{index}_{i}")
                
                elif line['type'] in [LineType.REF_SEGMENT.value, LineType.REF_CURVE.value]:
                    line['reference'] = st.text_input("Reference", line.get('reference') or '', key=f"ref_{index}_{i}")
                    line['monument'] = st.text_input("Monument", line.get('monument') or '', key=f"mon_{index}_{i}")
            
            if st.button(f"🗑️ Remove Line {i+1}", key=f"remove_{index}_{i}"):
                lines.pop(i)
                st.session_state[f'lines_{index}'] = lines
                st.rerun()
    
    # Save corrected entry
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save to Gold Dataset"):
            try:
                # Create corrected schema output
                corrected_lines = []
                for line_data in lines:
                    # Clean up None values and convert to proper types
                    clean_line_data = {}
                    for key, value in line_data.items():
                        if value == '' or value == 0:
                            clean_line_data[key] = None
                        else:
                            clean_line_data[key] = value
                    
                    corrected_lines.append(LineData(**clean_line_data))
                
                corrected_output = SchemaOutput(
                    bucket=corrected_bucket,
                    lines=corrected_lines
                )
                
                # Save to gold dataset
                if save_gold_entry(
                    text=text,
                    gold_output=corrected_output,
                    source_file=candidate.get('filename'),
                    verified_by=st.session_state.get('user', {}).get('email', 'manual_entry')
                ):
                    st.success("Saved to gold dataset!")
                    
                    # Clean up session state
                    if 'editing_candidate' in st.session_state:
                        del st.session_state.editing_candidate
                    if 'editing_index' in st.session_state:
                        del st.session_state.editing_index
                    if f'lines_{index}' in st.session_state:
                        del st.session_state[f'lines_{index}']
                    
                    # Remove from candidates
                    if 'candidates' in st.session_state and st.session_state.candidates:
                        if index < len(st.session_state.candidates):
                            st.session_state.candidates.pop(index)
                    
                    st.rerun()
                else:
                    st.error("Failed to save to gold dataset")
                    
            except Exception as e:
                st.error(f"Error saving to gold dataset: {e}")
    
    with col2:
        if st.button("❌ Cancel"):
            # Clean up session state
            if 'editing_candidate' in st.session_state:
                del st.session_state.editing_candidate
            if 'editing_index' in st.session_state:
                del st.session_state.editing_index
            if f'lines_{index}' in st.session_state:
                del st.session_state[f'lines_{index}']
            st.rerun()


def display_manual_entry_interface():
    """
    Interface for manual entry of gold dataset entries
    """
    st.subheader("✏️ Manual Gold Dataset Entry")
    
    # Text input
    text = st.text_area("Legal Description Text", height=150, 
                       placeholder="Enter the legal description text here...")
    
    source_file = st.text_input("Source File (optional)", 
                               placeholder="e.g., deed_book_123_page_456.pdf")
    
    if text:
        # Use the same editor interface as candidate editing
        # Create a mock candidate
        mock_candidate = {
            'text': text,
            'classification': 'no_bearings',
            'filename': source_file or 'manual_entry'
        }
        
        display_candidate_editor(mock_candidate, 'manual')


def display_all_entries_interface():
    """
    Interface for viewing and managing all gold dataset entries
    """
    st.subheader("📊 All Gold Dataset Entries")
    
    from utils.gold_dataset import load_all_gold_entries
    
    entries = load_all_gold_entries()
    
    if not entries:
        st.info("No gold dataset entries found.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bucket_filter = st.selectbox(
            "Filter by Bucket",
            ["All"] + [bc.value for bc in BucketClassification]
        )
    
    with col2:
        source_filter = st.selectbox(
            "Filter by Source",
            ["All"] + list(set([entry.get('source_file', 'Unknown') for entry in entries]))
        )
    
    with col3:
        limit = st.number_input("Limit Results", min_value=10, max_value=100, value=50)
    
    # Apply filters
    filtered_entries = entries
    
    if bucket_filter != "All":
        filtered_entries = [e for e in filtered_entries 
                          if e.get('gold_output', {}).get('bucket') == bucket_filter]
    
    if source_filter != "All":
        filtered_entries = [e for e in filtered_entries 
                          if e.get('source_file') == source_filter]
    
    filtered_entries = filtered_entries[:limit]
    
    st.write(f"Showing {len(filtered_entries)} of {len(entries)} entries")
    
    # Display entries
    for i, entry in enumerate(filtered_entries):
        gold_output = entry.get('gold_output', {})
        
        with st.expander(f"Entry {i+1}: {entry.get('source_file', 'Unknown')} - {gold_output.get('bucket', 'Unknown')}"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Text:**")
                st.text_area("", entry.get('text', ''), height=100, key=f"view_text_{i}", disabled=True)
                
                st.write("**Metadata:**")
                st.write(f"ID: {entry.get('id', 'Unknown')}")
                st.write(f"Created: {entry.get('created_at', 'Unknown')}")
                st.write(f"Verified by: {entry.get('verified_by', 'Unknown')}")
            
            with col2:
                st.write("**Gold Output:**")
                st.json(gold_output)
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"✏️ Edit Entry {i+1}", key=f"edit_entry_{i}"):
                    st.session_state.editing_entry = entry
                    st.session_state.editing_entry_index = i
                    st.rerun()
            
            with col2:
                if st.button(f"🗑️ Delete Entry {i+1}", key=f"delete_entry_{i}"):
                    # Would implement delete functionality
                    st.warning("Delete functionality not implemented yet")


def add_candidate_to_gold(candidate: Dict[str, Any]) -> bool:
    """
    Add a candidate to gold dataset with minimal processing
    """
    try:
        text = candidate['text']
        classification = candidate.get('classification', 'no_bearings')
        
        # Create basic schema output based on classification
        if classification == 'explicit_bearings':
            gold_output = SchemaOutput(bucket='explicit_bearings', lines=[])
        elif classification == 'abstract_bearings':
            gold_output = SchemaOutput(bucket='abstract_bearings', lines=[])
        else:
            gold_output = SchemaOutput(bucket='no_bearings', lines=[])
        
        return save_gold_entry(
            text=text,
            gold_output=gold_output,
            source_file=candidate.get('filename'),
            verified_by='auto_import'
        )
        
    except Exception as e:
        st.error(f"Error adding candidate to gold dataset: {e}")
        return False