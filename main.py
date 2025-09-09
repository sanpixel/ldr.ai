from dotenv import load_dotenv
load_dotenv()

import os
print("API Key exists:", bool(os.environ.get("OPENAI_API_KEY")))

# Debug Configuration
DEBUG_MODE = False
AUTO_PROCESS_DEBUG = False  # Auto-process first PDF for testing data collection
if AUTO_PROCESS_DEBUG:
    print("AUTO-PROCESS DEBUG ENABLED - Will auto-process first local PDF")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import radians
import ezdxf
from io import BytesIO, StringIO
import pytesseract
from pdf2image import convert_from_path
import re
import tempfile
import os
import glob
from openai import OpenAI
import json
import io
from datetime import datetime
import requests
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.graphics import shapes
from reportlab.graphics.shapes import Drawing, Line, String, Circle

# Try to import FreeCAD, but don't fail if it's not available
FREECAD_AVAILABLE = False
try:
    import FreeCAD
    import Part
    FREECAD_AVAILABLE = True
except ImportError:
    pass

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_folder_id_from_share_link(share_link):
    """Extract Google Drive folder ID from a share link."""
    try:
        # Handle different Google Drive share link formats
        # Format 1: https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing
        # Format 2: https://drive.google.com/open?id=FOLDER_ID
        
        if 'folders/' in share_link:
            folder_id = share_link.split('folders/')[1].split('?')[0]
        elif 'id=' in share_link:
            folder_id = share_link.split('id=')[1].split('&')[0]
        else:
            return None
        
        return folder_id
    except Exception as e:
        st.error(f"Error parsing Google Drive link: {str(e)}")
        return None

def list_pdfs_from_google_drive(folder_id):
    """List PDF files from a public Google Drive folder."""
    try:
        # Google Drive API endpoint for listing files in a folder
        api_url = f"https://www.googleapis.com/drive/v3/files"
        
        params = {
            'q': f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
            'fields': 'files(id,name,size,webContentLink)',
            'key': os.environ.get('GOOGLE_DRIVE_API_KEY')  # We'll need this API key
        }
        
        response = requests.get(api_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('files', [])
        else:
            st.error(f"Failed to access Google Drive folder: {response.status_code}")
            if response.status_code == 403:
                st.error("Folder may not be public or API key may be invalid")
            return []
            
    except Exception as e:
        st.error(f"Error listing Google Drive files: {str(e)}")
        return []

def download_pdf_from_google_drive(file_id):
    """Download a PDF file from Google Drive."""
    try:
        # First try: Direct public download URL (works for public files without auth)
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        response = requests.get(download_url)
        
        if response.status_code == 200:
            # Check if we got the actual PDF file
            if response.headers.get('content-type', '').startswith('application/pdf'):
                return BytesIO(response.content)
            elif len(response.content) > 1000:  # Likely got the PDF even without proper content-type
                return BytesIO(response.content)
        
        # Fallback: Try API method with key parameter (correct way for API key)
        if os.environ.get('GOOGLE_DRIVE_API_KEY'):
            api_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={os.environ.get('GOOGLE_DRIVE_API_KEY')}"
            api_response = requests.get(api_url)
            
            if api_response.status_code == 200:
                return BytesIO(api_response.content)
            else:
                st.error(f"Failed to download file: {api_response.status_code}")
                if api_response.status_code == 403:
                    st.error("File may not be publicly accessible or API key may be invalid")
                elif api_response.status_code == 404:
                    st.error("File not found - it may have been moved or deleted")
                return None
        else:
            st.error("No Google Drive API key configured and direct download failed")
            return None
            
    except Exception as e:
        st.error(f"Error downloading file: {str(e)}")
        return None

def format_bearing_concise(bearing_desc):
    """Convert verbose bearing description to concise surveyor's notation."""
    try:
        # Extract bearing components from the original text
        pattern = r'(North|South)\s+(\d+)\s*(?:°|degrees?|deg|\s)\s*(\d+)\s*(?:\'|′|minutes?|min|\s)\s*(?:(\d+)\s*(?:"|″|seconds?|sec|\s)\s+)?(East|West)'
        match = re.search(pattern, bearing_desc, re.IGNORECASE)

        if match:
            cardinal_ns, deg, min, sec, cardinal_ew = match.groups()
            # Format to concise notation
            ns = 'N' if cardinal_ns.lower() == 'north' else 'S'
            ew = 'E' if cardinal_ew.lower() == 'east' else 'W'
            sec = f" {int(sec):02d}s" if sec else ""
            return f"{ns} {deg}d {min}m{sec} {ew}"
    except Exception:
        return bearing_desc  # Return original if parsing fails
    return bearing_desc

def extract_bearings_with_gpt(text, filename, user_email, file_size=None, page_count=None, text_length=None, bearing_count=None):
    """Use GPT to extract bearings from text with a robust, line-by-line parser."""
    import time
    processing_start_time = time.time()
    
    try:
        # Load prompt from external file with robust fallback
        try:
            with open('bearings_prompt.txt', 'r', encoding='utf-8') as f:
                prompt_template = f.read().strip()
        except (FileNotFoundError, IOError, PermissionError) as e:
            if st.session_state.get('debug_enabled', False):
                st.warning(f"🔄 DEBUG: bearings_prompt.txt file not accessible ({str(e)}). Using embedded fallback prompt.")
            # Current working prompt as fallback (embedded for Cloud Run reliability)
            prompt_template = """First classify the legal description type using these criteria:

EXPLICIT_BEARINGS: Contains specific measurements like:
- "N 45° 30' 15\" E 150.00 feet"
- "North 71 degrees 53 minutes East 200 feet"
- "S 73° 32' 01\" W 125.50 feet"
- Any text with degrees, minutes, seconds AND distances

ABSTRACT_BEARINGS: Contains directional descriptions without specific measurements:
- "northerly along the creek line"
- "following the existing fence line"
- "along the property line of adjacent parcel"
- "easterly along the right-of-way"

EXTERNAL_REF: References external documents or survey systems:
- "Lot 5, Block 3, Happy Valley Subdivision"
- "as recorded in Plat Book 42, Page 15"
- "Section 12, Township 5 North, Range 3 West"
- "according to the plat thereof"

Provide your analysis in this exact format:

CLASSIFICATION: [explicit_bearings|abstract_bearings|external_ref]
CONFIDENCE: [high|medium|low]
REASONING: [explain why you chose this classification]
EVIDENCE: [all of the specific text that supports your decision]
RANK_ALTERNATIVES:
- Second choice: [classification] (reason)
- Third choice: [classification] (reason)

Then extract data based on classification:

If EXPLICIT_BEARINGS:

then Extract all bearings, distances, and monuments from the following legal description text. 
     For each line segment found, output in this exact format:
            BEARING: [bearing]
            DISTANCE: [distance]
            MONUMENT: [monument/reference]

            Ensure each attribute is on a new line.

If ABSTRACT_BEARINGS:
DESCRIPTION: [boundary description]
REFERENCE: [reference points/lines]

If EXTERNAL_REF:
LOT: [lot number]
BLOCK: [block number]
SUBDIVISION: [subdivision name]
PLAT_BOOK: [plat book reference]
SECTION: [section/township/range if applicable]

Text to analyze:"""
        
        prompt = prompt_template + "\n" + text
        
        # Original hardcoded prompt (commented out for reference)
        # prompt = """Extract all bearings, distances, and monuments from the following legal description text. 
        # For each line segment found, output in this exact format:
        # BEARING: [bearing]
        # DISTANCE: [distance]
        # MONUMENT: [monument/reference]
        #
        # Ensure each attribute is on a new line.
        # Text to analyze:
        # """ + text

        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:personal:ldr:BEoe3v67",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        result_text = response.choices[0].message.content
        
        # Parse reasoning data - will add filename first at the end
        reasoning_data = {
            'timestamp': datetime.now().isoformat(),
            'input_text': text[:1000],  # First 1000 chars for logging
            'full_response': result_text
        }
        
        # Extract reasoning components
        lines = result_text.split('\n')
        alternatives_lines = []
        collecting_alternatives = False
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('CLASSIFICATION:'):
                reasoning_data['classification'] = line.split(':', 1)[1].strip().lower()
            elif line.upper().startswith('CONFIDENCE:'):
                reasoning_data['confidence'] = line.split(':', 1)[1].strip().lower()
            elif line.upper().startswith('REASONING:'):
                reasoning_data['reasoning'] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('EVIDENCE:'):
                reasoning_data['evidence'] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('RANK_ALTERNATIVES:'):
                collecting_alternatives = True
                alternatives_lines = []
            elif collecting_alternatives and (line.startswith('- ') or line.strip().startswith('- ')):
                alternatives_lines.append(line.strip())
            elif collecting_alternatives and line and not line.strip().startswith('- ') and line.strip() and not line.strip().startswith('Second choice:') and not line.strip().startswith('Third choice:'):
                # Stop collecting if we hit a non-alternative line (but allow empty lines)
                collecting_alternatives = False
                reasoning_data['alternatives'] = '\n'.join(alternatives_lines)
        
        # Handle case where alternatives are at the end
        if collecting_alternatives and alternatives_lines:
            reasoning_data['alternatives'] = '\n'.join(alternatives_lines)
        
        # Add filename and user email first to ensure they appear first in JSON
        user_email = st.session_state.get('user', {}).get('email', 'anonymous')
        
        if st.session_state.get('debug_enabled', False):
            st.write(f"🔍 DEBUG: Before reordering - keys: {list(reasoning_data.keys())}")
            st.write(f"🔍 DEBUG: user_email: '{user_email}'")
            st.write(f"🔍 DEBUG: filename: '{filename if filename else 'Unknown'}'")
        
        # Create new ordered dictionary with filename and user_email first
        ordered_reasoning_data = {
            'filename': filename if filename else 'Unknown',
            'user_email': user_email
        }
        
        # Add metrics
        ordered_reasoning_data['file_size'] = file_size
        ordered_reasoning_data['debug_mode'] = DEBUG_MODE
        ordered_reasoning_data['processing_time'] = round(time.time() - processing_start_time, 3)
        ordered_reasoning_data['page_count'] = page_count
        ordered_reasoning_data['text_length'] = len(text) if text else 0
        ordered_reasoning_data['bearing_count'] = len([line for line in result_text.split('\n') if line.strip().upper().startswith('BEARING:')])
        ordered_reasoning_data['ocr_confidence'] = 0.85  # Placeholder - OCR libraries don't always provide confidence scores
        ordered_reasoning_data['model_version'] = "ft:gpt-3.5-turbo-0125:personal:ldr:BEoe3v67"
        ordered_reasoning_data['supplemental_info_found'] = bool(st.session_state.get('supplemental_info'))  # Boolean: found Land Lot/County data?
        
        # Calculate file hash for duplicate detection
        import hashlib
        if text:
            file_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        else:
            file_hash = None
        ordered_reasoning_data['file_hash'] = file_hash
        ordered_reasoning_data['original_filename'] = filename if filename else 'Unknown'  # User's original filename
        ordered_reasoning_data['storage_path'] = f"temp/{filename}" if filename else None  # Where file is stored
        
        # Determine upload method based on context
        if hasattr(st.session_state, 'drive_files') and st.session_state.drive_files:
            upload_method = "google_drive"
        elif filename and any(pdf_file == filename for pdf_file in glob.glob("*.pdf")):
            upload_method = "local_file"
        else:
            upload_method = "file_upload"
        ordered_reasoning_data['upload_method'] = upload_method  # How file was uploaded
        
        # Generate session ID for tracking
        import uuid
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        ordered_reasoning_data['session_id'] = st.session_state.session_id  # Browser session tracking
        
        # Get user IP address (proper Streamlit method)
        try:
            # Try to get IP from Streamlit context or headers
            ctx = st.runtime.scriptrunner.get_script_run_ctx()
            if ctx and hasattr(ctx, 'session_id'):
                # Try to get from session info or request headers
                ip_address = getattr(ctx, 'client_ip', None) or '127.0.0.1'
            else:
                ip_address = '127.0.0.1'
        except:
            ip_address = '127.0.0.1'
        ordered_reasoning_data['ip_address'] = ip_address  # User's IP for analytics
        
        # Get user agent from browser
        try:
            user_agent = st.context.headers.get('User-Agent', 'Unknown')
        except:
            user_agent = 'Unknown'
        ordered_reasoning_data['user_agent'] = user_agent  # Browser info
        
        # Track which prompt version was used
        ordered_reasoning_data['prompt_version'] = 'bearings_prompt.txt'  # Which prompt template was used
        
        # GPT settings used
        ordered_reasoning_data['temperature'] = 0.1  # GPT temperature setting
        ordered_reasoning_data['max_tokens'] = None  # Token limits (None = default)
        
        # Add all other fields in their original order
        for key, value in reasoning_data.items():
            ordered_reasoning_data[key] = value
        
        reasoning_data = ordered_reasoning_data
        
        # Display reasoning in debug mode (no expander to avoid nesting)
        if st.session_state.get('debug_enabled', False):
            st.write("🔍 **DEBUG: AI Classification Reasoning**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Classification**: {reasoning_data.get('classification', 'Not found')}")
                st.write(f"**Confidence**: {reasoning_data.get('confidence', 'Not found')}")
                if reasoning_data.get('alternatives'):
                    st.write(f"**Alternatives**:")
                    st.text(reasoning_data.get('alternatives'))
            with col2:
                st.write(f"**Reasoning**: {reasoning_data.get('reasoning', 'Not found')}")
                st.write(f"**Evidence**: {reasoning_data.get('evidence', 'Not found')}")
        
        # Parse bearings - GPT prompt handles classification, we just extract the data
        bearings = []
        current_bearing = {}
        
        # Process all lines to extract bearings (GPT already filtered by classification)
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith('BEARING:'):
                # Save previous bearing if it's complete
                if current_bearing.get('distance'):
                    bearings.append(current_bearing)
                
                bearing_text = line.split(':', 1)[1].strip()
                current_bearing = {'bearing': bearing_text} 

                # Try unified pattern for both formats
                # Handles: S 73° 32' 01" W AND North 71 degrees 51 minutes 19 seconds East
                pattern = r'(S|South|N|North)[\s\.]*(\d+)(?:[\s°degrees]+(?:(\d+)(?:[\s\'minutes]+(?:(\d+(?:\.\d+)?)(?:[\s"seconds]+)?)?)?)?)?[\s]*(E|W|East|West)'
                match = re.search(pattern, bearing_text, re.IGNORECASE)
                
                # Try long format: North 71 degrees 53 minutes 10 seconds East
                long_pattern = r'(North|South)\s+(\d+)\s+degrees?\s+(\d+)\s+minutes?\s+(\d+(?:\.\d+)?)\s+seconds?\s+(East|West)'
                long_match = re.search(long_pattern, bearing_text, re.IGNORECASE)
                
                if match:
                    groups = match.groups()
                    if st.session_state.get('debug_enabled', False):
                        st.markdown(f"<small>🔍 DEBUG: Successfully matched '{bearing_text}' | Groups: {groups} | Pattern: Standard</small>", unsafe_allow_html=True)
                    
                    ns_raw = (groups[0] or '').upper()
                    ew_raw = (groups[4] or '').upper()

                    current_bearing['cardinal_ns'] = 'South' if 'S' in ns_raw else 'North'
                    current_bearing['cardinal_ew'] = 'East' if 'E' in ew_raw else 'West'
                    current_bearing['degrees'] = int(groups[1]) if groups[1] else 0
                    current_bearing['minutes'] = int(groups[2]) if groups[2] else 0
                    current_bearing['seconds'] = int(float(groups[3])) if groups[3] else 0
                    current_bearing['original_text'] = bearing_text
                    
                elif long_match:
                    groups = long_match.groups()
                    if st.session_state.get('debug_enabled', False):
                        st.markdown(f"<small>🔍 DEBUG: Successfully matched LONG '{bearing_text}' | Groups: {groups} | Pattern: Long</small>", unsafe_allow_html=True)
                    
                    current_bearing['cardinal_ns'] = groups[0]  # North or South
                    current_bearing['cardinal_ew'] = groups[4]  # East or West
                    current_bearing['degrees'] = int(groups[1])
                    current_bearing['minutes'] = int(groups[2])
                    current_bearing['seconds'] = int(float(groups[3]))
                    current_bearing['original_text'] = bearing_text
                    
                else:
                    current_bearing['original_text'] = bearing_text  # Store even if parsing failed

            elif line.upper().startswith('DISTANCE:') and current_bearing:
                distance_text = line.split(':', 1)[1].strip()
                distance_match = re.search(r'(\d+(?:\.\d+)?)', distance_text)
                if distance_match:
                    current_bearing['distance'] = float(distance_match.group(1))

            elif line.upper().startswith('MONUMENT:') and current_bearing:
                current_bearing['monument'] = line.split(':', 1)[1].strip()

        # Add the last bearing if complete
        if current_bearing.get('distance'):
            bearings.append(current_bearing)

        # Return only fully parsed bearings, but let GPT handle the classification logic
        parsed_bearings = [b for b in bearings if 'cardinal_ns' in b]
        
        # Calculate parsing metrics after we have the results
        total_bearings = len(bearings)
        parsed_count = len(parsed_bearings)
        parsing_success_rate = round((parsed_count / total_bearings * 100) if total_bearings > 0 else 0, 2)
        
        # Add parsing metrics to reasoning data
        reasoning_data['parsed_bearing_count'] = parsed_count
        reasoning_data['parsing_success_rate'] = parsing_success_rate
        
        if st.session_state.get('debug_enabled', False):
            st.write(f"🔍 DEBUG: After adding parsing metrics - keys: {list(reasoning_data.keys())}")
            st.write(f"🔍 DEBUG: Final reasoning_data: {reasoning_data}")
        
        # Save reasoning data to database after all metrics are added
        try:
            from utils.classification import save_classification_data
            if save_classification_data(reasoning_data):
                if st.session_state.get('debug_enabled', False):
                    st.success("✅ Classification data saved to database")
            else:
                if st.session_state.get('debug_enabled', False):
                    st.warning("⚠️ Failed to save classification data to database")
        except Exception as log_error:
            st.warning(f"Could not save reasoning data to database: {str(log_error)}")
        
        if st.session_state.get('debug_enabled', False):
            classification = reasoning_data.get('classification', 'Unknown')
            st.write(f"**DEBUG: Classification: '{classification}' | Found {len(parsed_bearings)} valid bearings**")
            
            # Only show warning if there's an actual mismatch
            if classification.lower() not in ['explicit_bearings'] and len(parsed_bearings) > 0:
                st.write("**DEBUG: Unexpected - Non-explicit classification but found bearings. This might indicate prompt issues.**")
            elif classification.lower() in ['explicit_bearings'] and len(parsed_bearings) == 0:
                st.write("**DEBUG: Unexpected - Explicit classification but no bearings found. Check parsing logic.**")

        return parsed_bearings, result_text

    except Exception as e:
        st.error(f"Error using GPT to parse text: {str(e)}")
        return [], None

def extract_bearings_from_text(text):
    """
    This function is now deprecated. Legal descriptions and bearings are handled by GPT.
    Returns an empty list to maintain compatibility with existing code.
    """
    st.info("Pattern matching has been removed. Using GPT for legal description analysis.")
    return []

def dms_to_decimal(degrees, minutes, seconds, cardinal_ns, cardinal_ew):
    """Convert DMS (Degrees, Minutes, Seconds) to decimal degrees (azimuth)."""
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600

    # Convert surveyor's bearing to azimuth (clockwise from north)
    if cardinal_ns == 'North' and cardinal_ew == 'East':
        azimuth = decimal
    elif cardinal_ns == 'North' and cardinal_ew == 'West':
        azimuth = 360 - decimal
    elif cardinal_ns == 'South' and cardinal_ew == 'East':
        azimuth = 180 - decimal
    else:  # South and West
        azimuth = 180 + decimal

    return azimuth % 360

def decimal_to_dms(decimal_degrees):
    """Convert decimal degrees back to DMS format."""
    # Convert bearing to compass reading
    if 0 <= decimal_degrees <= 90:
        cardinal_ns = 'North'
        cardinal_ew = 'East'
        angle = 90 - decimal_degrees
    elif 90 < decimal_degrees <= 180:
        cardinal_ns = 'South'
        cardinal_ew = 'East'
        angle = decimal_degrees - 90
    elif 180 < decimal_degrees <= 270:
        cardinal_ns = 'South'
        cardinal_ew = 'West'
        angle = 270 - decimal_degrees
    else:
        cardinal_ns = 'North'
        cardinal_ew = 'West'
        angle = decimal_degrees - 270

    degrees = int(angle)
    minutes_float = (angle - degrees) * 60
    minutes = int(minutes_float)
    seconds = int((minutes_float - minutes) * 60)

    return cardinal_ns, degrees, minutes, seconds, cardinal_ew

def calculate_endpoint(start_point, bearing, distance):
    """Calculate endpoint coordinates given start point, bearing and distance."""
    bearing_rad = radians(bearing)
    dx = distance * np.sin(bearing_rad)
    dy = distance * np.cos(bearing_rad)
    end_x = start_point[0] + dx
    end_y = start_point[1] + dy
    return [end_x, end_y]

def create_dxf():
    """Create a DXF file from the current lines."""
    if st.session_state.lines.empty:
        st.error("No lines to export")
        return None

    try:
        # Create new document
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()

        # Add POB text and arrow
        try:
            # Add POB text
            msp.add_text(
                "POB",
                dxfattribs={
                    "height": 1.0,
                    "insert": (3, -3)  # Offset from origin
                }
            )
            # Add arrow to POB
            msp.add_line(
                (3, -3),  # Start at text location
                (0, 0),   # End at origin
            )
            # Add circle at POB (origin)
            msp.add_circle((0, 0), radius=0.5)
        except Exception as pob_error:
            st.warning(f"Error adding POB annotation: {str(pob_error)}")

        # Add each line
        for idx, row in st.session_state.lines.iterrows():
            start = (float(row['start_x']), float(row['start_y']))
            end = (float(row['end_x']), float(row['end_y']))

            try:
                # Add the line
                msp.add_line(start, end)

                # Add circles at start and end points
                msp.add_circle(start, radius=0.5)
                msp.add_circle(end, radius=0.5)

                # Calculate line angle for aligned dimension
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                angle = np.arctan2(dy, dx)

                # Calculate dimension line offset perpendicular to the bearing line
                offset_distance = 2.0  # Adjust this value to control dimension text placement
                offset_x = offset_distance * np.sin(angle)
                offset_y = -offset_distance * np.cos(angle)

                # Add dimension with basic parameters
                dim = msp.add_linear_dim(
                    base=(0, 0),  # Base point for dimension line
                    p1=start,     # Start point
                    p2=end,       # End point
                    text=f"{row['distance']:.2f}'"  # Distance text
                )

                # Add monument text at start point (from previous line's end)
                if idx > 0:  # For all points except POB
                    prev_row = st.session_state.lines.iloc[idx-1]
                    if 'monument' in prev_row and prev_row['monument']:
                        msp.add_text(
                            prev_row['monument'],
                            dxfattribs={
                                "height": 33,
                                "insert": (start[0] + 1, start[1] + 1),
                                "rotation": np.degrees(angle)  # Align text with line
                            }
                        )
            except Exception as line_error:
                st.error(f"Error adding line {idx+1}: {str(line_error)}")
                return None

        # Save the file
        filename = "line_drawing.dxf"
        doc.saveas(filename)
        st.write(f"DXF file saved as: {filename}")

        # Read the file back for download
        with open(filename, 'rb') as f:
            return f.read()

    except Exception as e:
        st.error(f"DXF creation error: {str(e)}")
        return None

def create_test_dxf():
    """Create a test DXF file with simple content."""
    try:
        st.write("Creating test DXF file...")
        # Create new document with setup=True
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()

        # Add test line with only X,Y coordinates
        start_x, start_y = 0, 0
        end_x, end_y = 10, 10
        st.write(f"Adding test line: ({start_x}, {start_y}) to ({end_x}, {end_y})")

        try:
            msp.add_line((start_x, start_y), (end_x, end_y), dxfattribs={"layer": "TestLayer"})
        except Exception as line_error:
            st.error(f"Error adding test line: {str(line_error)}")
            return None

        # Save the file
        filename = "test.dxf"
        doc.saveas(filename)
        st.write(f"Test DXF file saved as: {filename}")

        # Read the file back for download
        with open(filename, 'rb') as f:
            return f.read()

    except Exception as e:
        st.error(f"Test DXF creation error: {str(e)}")
        return None

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'lines' not in st.session_state:
        st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'bearing_desc', 'distance', 'monument'])
    if 'current_point' not in st.session_state:
        st.session_state.current_point = [0, 0]
    if 'gpt_analysis' not in st.session_state:
        st.session_state.gpt_analysis = None
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = None
    if 'parsed_bearings' not in st.session_state:
        st.session_state.parsed_bearings = None
    if 'pdf_image' not in st.session_state:
        st.session_state.pdf_image = None
    if 'supplemental_info' not in st.session_state:
        st.session_state.supplemental_info = None
    if 'manual_bearing' not in st.session_state:
        st.session_state.manual_bearing = None
    if 'line_count' not in st.session_state:
        st.session_state.line_count = 4  # Start with 4 lines by default
    if 'draw_lines_section_expanded' not in st.session_state:
        st.session_state.draw_lines_section_expanded = False  # Start collapsed by default
    
    # Initialize the keys for the input fields
    for i in range(20): # Initialize for a max of 20 lines
        if f'cardinal_ns_{i}' not in st.session_state:
            st.session_state[f'cardinal_ns_{i}'] = "North"
            st.session_state[f'degrees_{i}'] = 0
            st.session_state[f'minutes_{i}'] = 0
            st.session_state[f'seconds_{i}'] = 0
            st.session_state[f'cardinal_ew_{i}'] = "East"
            st.session_state[f'distance_{i}'] = 0.0
            st.session_state[f'monument_{i}'] = ""

def draw_lines():
    """Create a Plotly figure with all lines."""
    fig = go.Figure()

    # Add POB annotation at origin
    fig.add_annotation(
        x=0,
        y=0,
        text="POB",
        showarrow=True,
        arrowhead=2,
        ax=30,  # Offset x position for text
        ay=-30,  # Offset y position for text
        font=dict(size=14),
        arrowsize=1.5,
        arrowwidth=2
    )

    # Draw all lines
    for idx, row in st.session_state.lines.iterrows():
        # Calculate midpoint for text position
        mid_x = (row['start_x'] + row['end_x']) / 2
        mid_y = (row['start_y'] + row['end_y']) / 2

        # Add line
        fig.add_trace(go.Scatter(
            x=[row['start_x'], row['end_x']],
            y=[row['start_y'], row['end_y']],
            mode='lines',
            name=f'Line {idx+1}',
            line=dict(width=2)
        ))

        # Add text label with concise bearing format
        bearing_text = format_bearing_concise(row["bearing_desc"])
        fig.add_trace(go.Scatter(
            x=[mid_x],
            y=[mid_y],
            mode='text',
            text=[f"{bearing_text}<br>{row['distance']:.2f} ft"],
            textposition='top center',
            hoverinfo='text',
            showlegend=False
        ))

        # Add points
        fig.add_trace(go.Scatter(
            x=[row['start_x']],
            y=[row['start_y']],
            mode='markers',
            name=f'Point {idx}',
            marker=dict(size=8)
        ))

    # Add final point
    if not st.session_state.lines.empty:
        fig.add_trace(go.Scatter(
            x=[st.session_state.lines.iloc[-1]['end_x']],
            y=[st.session_state.lines.iloc[-1]['end_y']],
            mode='markers',
            name=f'Point {len(st.session_state.lines)}',
            marker=dict(size=8)
        ))

    # Update layout with both zoom and pan enabled
    fig.update_layout(
        showlegend=False,
        title='Line Drawing',
        xaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            scaleanchor="y",
            scaleratio=1,
            constrain="domain"
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            constrain="domain"
        ),
        width=800,
        height=600,
        dragmode='pan'  # Enable panning by default
    )

    # Configure modebar with both zoom and pan options
    fig.update_layout(
        modebar=dict(
            add=['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
        )
    )

    return fig

def create_rectangle(start_point, side_length):
    """Create a rectangle-like shape with slightly randomized angles that closes back to start."""
    lines = []
    current = start_point
    first_point = start_point

    # Define three slightly randomized directions
    directions = [
        ("North", np.random.randint(0, 15), np.random.randint(0, 60), np.random.randint(0, 60), "East"),    # ~North
        ("North", np.random.randint(75, 90), np.random.randint(0, 60), np.random.randint(0, 60), "East"),   # ~East
        ("South", np.random.randint(0, 15), np.random.randint(0, 60), np.random.randint(0, 60), "East"),    # ~South
    ]

    # Draw first three lines with random angles
    for cardinal_ns, deg, min, sec, cardinal_ew in directions:
        # Convert DMS to decimal
        bearing = dms_to_decimal(deg, min, sec, cardinal_ns, cardinal_ew)

        # Calculate endpoint
        end_point = calculate_endpoint(current, bearing, side_length)

        # Create bearing description
        bearing_desc = f"{cardinal_ns} {deg}° {min}' {sec}\" {cardinal_ew}"

        # Add line using DataFrame
        new_line = pd.DataFrame({
            'start_x': [current[0]],
            'start_y': [current[1]],
            'end_x': [end_point[0]],
            'end_y': [end_point[1]],
            'bearing': [bearing],
            'bearing_desc': [bearing_desc],
            'distance': [side_length]
        })
        lines.append(new_line)

        current = end_point

    # Calculate the bearing and distance for the closing line
    dx = first_point[0] - current[0]
    dy = first_point[1] - current[1]
    closing_distance = np.sqrt(dx*dx + dy*dy)
    closing_bearing = np.degrees(np.arctan2(dx, dy)) % 360

    # Convert the closing bearing to DMS format
    cardinal_ns, deg, min, sec, cardinal_ew = decimal_to_dms(closing_bearing)
    bearing_desc = f"{cardinal_ns} {deg}° {min}' {sec}\" {cardinal_ew}"

    # Add the closing line
    new_line = pd.DataFrame({
        'start_x': [current[0]],
        'start_y': [current[1]],
        'end_x': [first_point[0]],
        'end_y': [first_point[1]],
        'bearing': [closing_bearing],
        'bearing_desc': [bearing_desc],
        'distance': [closing_distance]
    })
    lines.append(new_line)

    # Concatenate all lines into a single DataFrame
    return pd.concat(lines, ignore_index=True)

def extract_supplemental_info_with_gpt(text):
    """Use GPT to extract Land Lot #, District, and County information."""
    try:
        prompt = """Extract the Land Lot number, District, and County information from the following text.
        Format the response exactly like this example:
        Land Lot: 123
        District: 2nd
        County: Fulton

        Text to analyze:
        """ + text

        # Call GPT-4 with the prompt
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0
        )

        # Get the response text
        result_text = response.choices[0].message.content

        # Parse the response
        land_lot = None
        district = None
        county = None

        for line in result_text.split('\n'):
            if line.strip().startswith('Land Lot:'):
                land_lot = line.replace('Land Lot:', '').strip()
            elif line.strip().startswith('District:'):
                district = line.replace('District:', '').strip()
            elif line.strip().startswith('County:'):
                county = line.replace('County:', '').strip()

        return {'land_lot': land_lot, 'district': district, 'county': county}
    except Exception as e:
        st.error(f"Error extracting supplemental info: {str(e)}")
        return None

def draw_lines_from_bearings():
    """Draw lines using the parsed bearings from session state."""
    # Reset current point and create empty DataFrame with explicit dtypes
    st.session_state.current_point = [0, 0]
    st.session_state.lines = pd.DataFrame({
        'start_x': pd.Series(dtype='float64'),
        'start_y': pd.Series(dtype='float64'),
        'end_x': pd.Series(dtype='float64'),
        'end_y': pd.Series(dtype='float64'),
        'bearing': pd.Series(dtype='float64'),
        'bearing_desc': pd.Series(dtype='object'),
        'distance': pd.Series(dtype='float64'),
        'monument': pd.Series(dtype='object')
    })

    # If we have parsed bearings from PDF or manual input, process them
    bearings_to_process = st.session_state.parsed_bearings if st.session_state.parsed_bearings else []

    # Process each bearing
    for bearing in bearings_to_process:
        # Only process lines with non-zero distance
        distance = bearing['distance']
        if distance > 0:
            # Convert DMS to decimal degrees
            bearing_decimal = dms_to_decimal(
                bearing['degrees'],
                bearing['minutes'],
                bearing['seconds'],
                bearing['cardinal_ns'],
                bearing['cardinal_ew']
            )

            # Calculate new endpoint
            end_point = calculate_endpoint(st.session_state.current_point, bearing_decimal, distance)

            # Create bearing description
            bearing_desc = bearing['original_text']

            # Add new line to DataFrame with explicit dtypes
            new_line = pd.DataFrame({
                'start_x': [st.session_state.current_point[0]],
                'start_y': [st.session_state.current_point[1]],
                'end_x': [end_point[0]],
                'end_y': [end_point[1]],
                'bearing': [bearing_decimal],
                'bearing_desc': [bearing_desc],
                'distance': [distance],
                'monument': [bearing['monument']]
            }).astype({
                'start_x': 'float64',
                'start_y': 'float64',
                'end_x': 'float64',
                'end_y': 'float64',
                'bearing': 'float64',
                'bearing_desc': 'object',
                'distance': 'float64',
                'monument': 'object'
            })
            
            # Use concat with explicit dtypes
            st.session_state.lines = pd.concat([st.session_state.lines, new_line], ignore_index=True)

            # Update current point
            st.session_state.current_point = end_point

def process_pdf(uploaded_file):
    """Process uploaded PDF file and extract bearings."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        # Convert PDF to images
        images = convert_from_path(pdf_path)

        # Store the first page image in session state
        if images:
            # Convert PIL image to bytes for display
            img_byte_arr = BytesIO()
            images[0].save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            st.session_state.pdf_image = img_byte_arr

        # Extract text from each page
        extracted_text = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            extracted_text += f"\n--- Page {i+1} ---\n{text}\n"

        # Clean up temporary file
        os.unlink(pdf_path)

        # Store extracted text in session state
        st.session_state.extracted_text = extracted_text
        
        # Store extracted text and processing info in session state for display outside
        st.session_state.extracted_text = extracted_text
        st.session_state.processing_messages = []
        
        # Extract supplemental information first
        if os.environ.get("OPENAI_API_KEY"):
            try:
                supplemental_info = extract_supplemental_info_with_gpt(extracted_text)
                if supplemental_info:
                    st.session_state.supplemental_info = supplemental_info
                    st.success("Successfully extracted property information")
            except Exception as e:
                st.error(f"Error extracting property information: {str(e)}")

        # First try GPT extraction for bearings
        if os.environ.get("OPENAI_API_KEY"):
            try:
                # Get the filename, file size, and page count from the uploaded file
                filename = getattr(uploaded_file, 'name', 'Uploaded File')
                file_size = len(uploaded_file.getvalue())
                page_count = len(images)
                if st.session_state.get('debug_enabled', False):
                    st.write(f"🔍 DEBUG: Extracted filename: '{filename}' from uploaded file")
                    st.write(f"🔍 DEBUG: File size: {file_size} bytes")
                    st.write(f"🔍 DEBUG: Page count: {page_count} pages")
                bearings, result_text = extract_bearings_with_gpt(extracted_text, filename, st.session_state.get('user', {}).get('email', 'anonymous'), file_size, page_count)
                st.info(f"GPT returned {len(bearings)} bearings")
                
                # Store the GPT response for debug display
                st.session_state.gpt_response = result_text
                
                if bearings:
                    st.success(f"✅ Successfully extracted {len(bearings)} bearings!")
                    return bearings
                else:
                    st.warning("No bearings found")
                    return []
            except Exception as e:
                st.error(f"GPT analysis failed: {str(e)}")
                return []
        else:
            st.warning("No OpenAI API key found. Please configure your OpenAI API key to analyze legal descriptions.")
            return []
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

def export_cad():
    """Create a CAD file using FreeCAD."""
    if not FREECAD_AVAILABLE:
        st.error("FreeCAD is not available. Please use DXF export instead.")
        return None

    if st.session_state.lines.empty:
        st.error("No lines to export")
        return None

    try:
        # Create a new FreeCAD document
        doc = FreeCAD.newDocument("LineDrawing")

        # Add POB point
        pob = Part.makeVertex(0, 0, 0)
        pob_obj = doc.addObject("Part::Feature", "POB")
        pob_obj.Shape = pob

        # Add POB label
        label = doc.addObject("App::Annotation", "POB_Label")
        label.LabelText = "POB"
        label.Position = FreeCAD.Vector(3, -3, 0)

        # Add each line
        for idx, row in st.session_state.lines.iterrows():
            try:
                # Create line
                start = FreeCAD.Vector(float(row['start_x']), float(row['start_y']), 0)
                end = FreeCAD.Vector(float(row['end_x']), float(row['end_y']), 0)
                line = Part.LineSegment(start, end)

                # Add line to document
                line_obj = doc.addObject("Part::Feature", f"Line_{idx+1}")
                line_obj.Shape = Part.Shape([line])
                # Add dimension
                dim = doc.addObject("TechDraw::DrawViewDimension", f"Dimension_{idx+1}")
                dim.Type = "Distance"
                dim.X = (start.x + end.x) / 2
                dim.Y = (start.y + end.y) / 2
                dim.Text = f"{row['distance']:.2f}'"

                # Add monument text if available
                if 'monument' in row and row['monument']:
                    monument = doc.addObject("App::Annotation", f"Monument_{idx+1}")
                    monument.LabelText = row['monument']
                    monument.Position = FreeCAD.Vector(end.x + 1, end.y + 1, 0)

            except Exception as line_error:
                st.warning(f"Error adding line {idx+1}: {str(line_error)}")
                continue

        # Save the file
        filename = "line_drawing.FCStd"
        doc.saveAs(filename)

        # Read the file back for download
        with open(filename, 'rb') as f:
            return f.read()

    except Exception as e:
        st.error(f"CAD creation error: {str(e)}")
        return None


def export_pdf():
    """Create a PDF file containing the line drawing and property information."""
    if st.session_state.lines.empty:
        st.error("No lines to export")
        return None

    try:
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=12, leftMargin=12, topMargin=12, bottomMargin=12)
        # Create the story (content) for the PDF
        story = []
        styles = getSampleStyleSheet()

        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=2,
            alignment=TA_CENTER
        )
        story.append(Paragraph("Property Survey Report", title_style))

        # Add static note under title
        note_style = ParagraphStyle(
            'Note',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=2,
            alignment=TA_CENTER
        )
        story.append(Paragraph("Computer recognized bearings are referenced to the Georgia State Plane Coordinate System from provided legal description, please verify property lines with a licensed surveyor", note_style))

        # Add property information if available
        if st.session_state.supplemental_info:
            info_style = ParagraphStyle(
                'InfoStyle',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=2,
                alignment=TA_LEFT
            )

            # Create a table for property information
            info_data = [
                ["Land Lot:", str(st.session_state.supplemental_info.get('land_lot', 'N/A'))],
                ["District:", str(st.session_state.supplemental_info.get('district', 'N/A'))],
                ["County:", str(st.session_state.supplemental_info.get('county', 'N/A'))]
            ]

            info_table = Table(info_data, colWidths=[1.5*inch])
            info_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ]))
            story.append(info_table)
            story.append(Spacer(1, 20))

        # Create a drawing of the lines using ReportLab
        # Calculate the bounds of the drawing
        if not st.session_state.lines.empty:
            min_x = min(st.session_state.lines['start_x'].min(), st.session_state.lines['end_x'].min())
            max_x = max(st.session_state.lines['start_x'].max(), st.session_state.lines['end_x'].max())
            min_y = min(st.session_state.lines['start_y'].min(), st.session_state.lines['end_y'].min())
            max_y = max(st.session_state.lines['start_y'].max(), st.session_state.lines['end_y'].max())

            # Add padding
            padding = max((max_x - min_x), (max_y - min_y)) * 0.1
            min_x -= padding
            max_x += padding
            min_y -= padding
            max_y += padding

            # Create drawing with proper aspect ratio
            width = 400
            height = 400
            scale_x = width / (max_x - min_x) if max_x != min_x else 1
            scale_y = height / (max_y - min_y) if max_y != min_y else 1
            scale = min(scale_x, scale_y)

            d = Drawing(width + 5, height + 5)  # Add margins

            # Helper function to transform coordinates
            def transform_point(x, y):
                return (
                    25 + (x - min_x) * scale,
                    25 + (y - min_y) * scale
                )

            # Draw POB point and label
            pob_x, pob_y = transform_point(0, 0)
            d.add(Circle(pob_x, pob_y, 3, fillColor=colors.black))
            d.add(String(pob_x + 10, pob_y - 10, 'POB'))

            # Draw all lines
            for idx, row in st.session_state.lines.iterrows():
                start_x, start_y = transform_point(row['start_x'], row['start_y'])
                end_x, end_y = transform_point(row['end_x'], row['end_y'])

                # Draw line
                d.add(Line(start_x, start_y, end_x, end_y, strokeColor=colors.black, strokeWidth=1))

                # Add bearing text
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                bearing_text = format_bearing_concise(row['bearing_desc'])
                d.add(String(mid_x, mid_y + 10, bearing_text))
                d.add(String(mid_x, mid_y - 5, f"{row['distance']:.2f}'"))

            # Add the drawing to the story
            story.append(d)

        # Add bearing information
        if not st.session_state.lines.empty:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Survey Lines", styles['Heading2']))

            # Create table for bearings
            bearing_data = [["Line", "Bearing", "Distance", "Monument"]]
            for idx, row in st.session_state.lines.iterrows():
                bearing_data.append([
                    f"Line {idx + 1}",
                    format_bearing_concise(row['bearing_desc']),
                    f"{row['distance']:.2f}'",
                    row.get('monument', '')
                ])

            bearing_table = Table(bearing_data, colWidths=[1*inch, 2*inch, 1.5*inch, 2.5*inch])
            bearing_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E6E6E6')),  # Light gray background
            ]))
            story.append(bearing_table)

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        st.error(f"PDF creation error: {str(e)}")
        return None


def manual_bearing_input_to_parsed_format(cardinal_ns, degrees, minutes, seconds, cardinal_ew, distance, monument):
    """Convert manual input fields to parsed bearing format."""
    try:
        # Convert inputs to appropriate types
        degrees = int(degrees) if degrees is not None else 0
        minutes = int(minutes) if minutes is not None else 0
        seconds = int(seconds) if seconds is not None else 0
        distance = float(distance) if distance is not None else 0.00

        # Skip default value warning if values are already set
        if cardinal_ns and cardinal_ew and (degrees > 0 or minutes > 0 or seconds > 0 or distance > 0):
            return {
                'cardinal_ns': cardinal_ns,
                'degrees': degrees,
                'minutes': minutes,
                'seconds': seconds,
                'cardinal_ew': cardinal_ew,
                'distance': distance,
                'monument': monument if monument else "",
                'original_text': f"{cardinal_ns} {degrees}° {minutes}' {seconds}\" {cardinal_ew}, {distance} feet {monument}"
            }
        elif any([cardinal_ns, cardinal_ew, degrees, minutes, seconds, distance]):
            # Show warning only if some values are set but not all
            st.warning("Created bearing with default values. Please verify the input.")
            return {
                'cardinal_ns': cardinal_ns or "North",
                'degrees': degrees,
                'minutes': minutes,
                'seconds': seconds,
                'cardinal_ew': cardinal_ew or "East",
                'distance': distance,
                'monument': monument if monument else "",
                'original_text': f"{cardinal_ns or 'North'} {degrees}° {minutes}' {seconds}\" {cardinal_ew or 'East'}, {distance} feet {monument}"
            }
        return None
    except Exception as e:
        st.error(f"Error parsing manual input: {str(e)}")
        return None


def generate_random_bearing():
    """Generate random but realistic bearing values."""
    import random

    # Generate realistic bearing values
    cardinal_ns = random.choice(["North", "South"])
    degrees = random.randint(0, 89)  # Avoid 90 to keep it realistic
    minutes = random.randint(0, 59)
    seconds = random.randint(0, 59)
    cardinal_ew = random.choice(["East", "West"])
    distance = round(random.uniform(50.0, 500.0), 2)  # Realistic distances between 50-500 feet
    monuments = [
        "to an iron pin",
        "to a stone marker",
        "to a concrete monument",
        "to a fence post",
        "to a corner post",
        ""
    ]
    monument = random.choice(monuments)

    return {
        'cardinal_ns': cardinal_ns,
        'degrees': degrees,
        'minutes': minutes,
        'seconds': seconds,
        'cardinal_ew': cardinal_ew,
        'distance': distance,
        'monument': monument
    }

def show_video_intro():
    """Show 7-second video intro with fade transition to main app."""
    # Custom CSS for video intro and fade effect
    st.markdown("""
    <style>
    .video-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 9999;
        background: black;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .video-intro {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .fade-out {
        animation: fadeOut 1s ease-out forwards;
        animation-delay: 6s;
    }
    
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; visibility: hidden; }
    }
    
    .main-app {
        opacity: 0;
        animation: fadeIn 1s ease-in forwards;
        animation-delay: 7s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    
    <div class="video-container fade-out">
        <video class="video-intro" autoplay muted>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    
    <script>
    setTimeout(function() {
        document.querySelector('.video-container').style.display = 'none';
    }, 7000);
    </script>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide", page_title="Legal Description Reader")
    
    # Import auth utilities
    try:
        from utils.auth import get_current_user, show_user_menu, show_login_button, supabase
        from utils.st_local_storage import StLocalStorage
        
        # Handle OAuth callback
        st_ls = StLocalStorage()
        query_params = st.query_params
        
        if "code" in query_params:
            try:
                # Exchange code for session
                response = supabase.auth.exchange_code_for_session({"auth_code": query_params["code"]})
                if response.user:
                    # Store session in local storage
                    session_data = {
                        "access_token": response.session.access_token,
                        "refresh_token": response.session.refresh_token
                    }
                    st_ls.set("g_session", session_data)
                    st.session_state.user = response.user.user_metadata
                    st.success(f"Welcome! Signed in as {response.user.email}")
                    # Clear query params and reload
                    st.query_params.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"Login failed: {str(e)}")
        
        # Get current user (don't require login)
        user = get_current_user()
        
        # Show user menu in sidebar if authenticated
        if user:
            show_user_menu()
        
    except ImportError:
        st.warning("Authentication module not found. Login features disabled.")
        user = None
    except Exception as auth_error:
        st.warning(f"Authentication error: {str(auth_error)}. Login features disabled.")
        user = None
    
    # Initialize session state for intro
    if 'show_intro' not in st.session_state:
        st.session_state.show_intro = False  # Disabled for debugging
        st.session_state.intro_start_time = None
    
    # Show video intro on first load
    if st.session_state.show_intro:
        import base64
        import time
        
        # Set start time if not set
        if st.session_state.intro_start_time is None:
            st.session_state.intro_start_time = time.time()
        
        # Check if 7 seconds have passed
        if time.time() - st.session_state.intro_start_time >= 7:
            st.session_state.show_intro = False
            st.rerun()
        
        # Read and encode the video file
        try:
            with open('loading-video.mp4', 'rb') as video_file:
                video_bytes = video_file.read()
                video_base64 = base64.b64encode(video_bytes).decode()
            
            # Custom CSS and HTML for video intro
            st.markdown(f"""
            <style>
            .video-container {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                z-index: 9999;
                background: black;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            
            .video-intro {{
                width: 100%;
                height: 100%;
                object-fit: cover;
            }}
            
            .fade-out {{
                animation: fadeOut 1s ease-out forwards;
                animation-delay: 6s;
            }}
            
            @keyframes fadeOut {{
                from {{ opacity: 1; }}
                to {{ opacity: 0; visibility: hidden; }}
            }}
            </style>
            
            <div class="video-container fade-out" id="videoIntro">
                <video class="video-intro" autoplay muted>
                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            
            <script>
            setTimeout(function() {{
                window.location.reload();
            }}, 7000);
            </script>
            """, unsafe_allow_html=True)
            
            # Auto-refresh every second to check timing
            st.rerun()
            
        except FileNotFoundError:
            st.warning("Video file 'loading-video.mp4' not found. Proceeding to main application.")
            st.session_state.show_intro = False
            st.rerun()
        
        # Don't show anything else during intro
        return
    
    # Main application (shown after intro)
    st.title("Legal Description Reader")
    
    # Debug toggle in sidebar
    with st.sidebar:
        st.subheader("Debug Controls")
        debug_password = st.text_input("Debug Password", type="password", key="debug_pw")
        if debug_password == "warez":
            st.session_state.debug_enabled = True
            st.success("🔍 Debug mode enabled")
        elif debug_password and debug_password != "warez":
            st.session_state.debug_enabled = False
            st.error("❌ Invalid password")
        
        if st.session_state.get('debug_enabled', False):
            st.info("🐛 Debug output active")
    
    initialize_session_state()
    
    # Custom CSS for all buttons - moved to top so it applies to all buttons
    st.markdown("""
    <style>
    /* Green primary button for all Process PDF buttons */
    .stButton > button[kind="primary"] {
        background-color: #28a745 !important;
        border-color: #28a745 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
    }
    
    /* Light blue secondary button for other buttons */
    .stButton > button[kind="secondary"] {
        background-color: #17a2b8 !important;
        border-color: #17a2b8 !important;
        color: white !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #138496 !important;
        border-color: #117a8b !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create two columns for the main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # PDF Upload Section
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        if uploaded_file is not None:
            if st.button("Process PDF", type="primary"):
                st.info("🔄 Processing PDF...")
                bearings = process_pdf(uploaded_file)
                if bearings:
                    st.session_state.parsed_bearings = bearings
                    st.session_state.line_count = len(bearings)
                    for i, bearing in enumerate(bearings):
                        st.session_state[f"cardinal_ns_{i}"] = bearing.get('cardinal_ns', "North")
                        st.session_state[f"degrees_{i}"] = bearing.get('degrees', 0)
                        st.session_state[f"minutes_{i}"] = bearing.get('minutes', 0)
                        st.session_state[f"seconds_{i}"] = bearing.get('seconds', 0)
                        st.session_state[f"cardinal_ew_{i}"] = bearing.get('cardinal_ew', "East")
                        st.session_state[f"distance_{i}"] = float(bearing.get('distance', 0.0))
                        st.session_state[f"monument_{i}"] = bearing.get('monument', '')
                    
                    st.session_state.draw_lines_section_expanded = False
                    st.success(f"✅ Successfully extracted and populated {len(bearings)} bearings!")
        
        # Available PDF Files Selector
        import glob
        pdf_files = glob.glob("*.pdf")
        
        st.subheader("📄 Example PDFs")
        
        # Create tabs for different sources
        tab1, tab2 = st.tabs(["📂 Local Files", "☁️ Google Drive"])
        
        with tab1:
            if pdf_files:
                # AUTO-PROCESS first PDF for testing (only when AUTO_PROCESS_DEBUG is enabled)
                if AUTO_PROCESS_DEBUG and 'auto_processed' not in st.session_state:
                    st.session_state.auto_processed = True
                    first_pdf = pdf_files[0]
                    st.info(f"🤖 AUTO-PROCESSING: {first_pdf} for data collection testing...")
                    
                    # Use the same logic as the manual "Process PDF" button
                    try:
                        with open(first_pdf, "rb") as pdf_file:
                            file_content = pdf_file.read()
                        
                        with st.spinner(f'AUTO-PROCESSING {first_pdf}...'):
                            from io import BytesIO
                            pdf_buffer = BytesIO(file_content)
                            pdf_buffer.name = first_pdf
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
                                
                                st.session_state.draw_lines_section_expanded = False
                                st.success(f"✅ AUTO-PROCESSED: Extracted {len(bearings)} bearings from {first_pdf}!")
                            else:
                                st.warning(f"⚠️ AUTO-PROCESS: No bearings found in {first_pdf}")
                                
                    except Exception as e:
                        st.error(f"❌ AUTO-PROCESS ERROR: {str(e)}")
                # Create dropdown selector
                selected_pdf = st.selectbox(
                    "Choose a PDF file to process:",
                    options=pdf_files,
                    index=0
                )
                
                if selected_pdf:
                    # Display file info with caption
                    try:
                        import os
                        file_size = os.path.getsize(selected_pdf)
                        file_size_kb = file_size / 1024
                        if file_size_kb > 1024:
                            size_display = f"{file_size_kb/1024:.1f} MB"
                        else:
                            size_display = f"{file_size_kb:.1f} KB"
                        st.caption(f"Selected: {selected_pdf} ({size_display})")
                    except Exception:
                        st.caption(f"Selected: {selected_pdf}")
                    
                    # Process button
                    if st.button(f"🔄 Process {selected_pdf}", use_container_width=True, type="primary"):
                        try:
                            with open(selected_pdf, "rb") as pdf_file:
                                file_content = pdf_file.read()
                            
                            with st.spinner(f'Processing {selected_pdf}...'):
                                # Create a BytesIO object to simulate uploaded file
                                from io import BytesIO
                                pdf_buffer = BytesIO(file_content)
                                # Set the name attribute so we can get the actual filename
                                pdf_buffer.name = selected_pdf
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
                                    
                                    st.session_state.draw_lines_section_expanded = False
                                    st.success(f"✅ Extracted {len(bearings)} bearings from {selected_pdf}!")
                                else:
                                    st.warning("⚠️ No bearings found.")
                                    
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            else:
                st.info("No PDF files found in the project directory.")
        
        with tab2:
                # Input for Google Drive folder link
                drive_link = st.text_input(
                    "Google Drive folder share link:",
                    value="https://drive.google.com/drive/folders/1-BiLAKzEGndi3XQAxcAD1zQ-McKHodgS?usp=drive_link",
                    help="Make sure the folder is publicly shared (anyone with link can view)"
                )
                
                if drive_link:
                    folder_id = extract_folder_id_from_share_link(drive_link)
                    
                    if folder_id:
                        if st.button("🔍 List PDFs from Google Drive", type="secondary"):
                            if not os.environ.get('GOOGLE_DRIVE_API_KEY'):
                                st.error("Google Drive API key not configured. Please add GOOGLE_DRIVE_API_KEY to environment variables.")
                            else:
                                with st.spinner("Fetching PDFs from Google Drive..."):
                                    drive_files = list_pdfs_from_google_drive(folder_id)
                                    
                                    if drive_files:
                                        st.success(f"Found {len(drive_files)} PDF files")
                                        # Store the files in session state
                                        st.session_state.drive_files = drive_files
                                    else:
                                        st.warning("No PDF files found in the Google Drive folder or folder is not accessible.")
                        
                        # Show file selection and process button if files are loaded
                        if hasattr(st.session_state, 'drive_files') and st.session_state.drive_files:
                            st.info(f"📁 Loaded: {len(st.session_state.drive_files)} files from Google Drive")
                            
                            # Display the files
                            selected_drive_file = st.selectbox(
                                "Choose a PDF from Google Drive:",
                                options=[f["name"] for f in st.session_state.drive_files],
                                format_func=lambda x: f"{x} ({next(f['size'] for f in st.session_state.drive_files if f['name'] == x) if 'size' in st.session_state.drive_files[0] else 'Unknown size'} bytes)"
                            )
                            
                            if selected_drive_file:
                                # Find the selected file
                                selected_file_data = next(f for f in st.session_state.drive_files if f["name"] == selected_drive_file)
                                
                                if st.button(f"🔄 Process {selected_drive_file}", use_container_width=True, type="primary", key="drive_process"):
                                    with st.spinner(f"Downloading and processing {selected_drive_file}..."):
                                        # Download the file
                                        pdf_buffer = download_pdf_from_google_drive(selected_file_data["id"])
                                        
                                        if pdf_buffer:
                                            # Process the PDF
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
                                                
                                                st.session_state.draw_lines_section_expanded = False
                                                st.success(f"✅ Extracted {len(bearings)} bearings from {selected_drive_file}!")
                                            else:
                                                st.warning("⚠️ No bearings found.")
                                        else:
                                            st.error("❌ Failed to download file from Google Drive")
                    else:
                        st.error("❌ Invalid Google Drive link. Please check the format.")
                else:
                    st.info("💡 To use Google Drive integration:")
                    st.markdown("""
                    1. Go to your Google Drive folder
                    2. Right-click and select "Share"
                    3. Set permissions to "Anyone with the link can view"
                    4. Copy the link and paste it above
                    """)

    with col2:
        # Show login button if not authenticated
        try:
            if not user:
                show_login_button()
        except:
            pass  # Ignore auth errors in this section
            
        if st.session_state.pdf_image:
            st.image(st.session_state.pdf_image, caption='PDF Preview', use_container_width=True)

    # Display processing messages if available (from PDF processing)
    if hasattr(st.session_state, 'processing_messages') and st.session_state.processing_messages:
        for msg_type, msg_text in st.session_state.processing_messages:
            if msg_type == "success":
                st.success(msg_text)
            elif msg_type == "error":
                st.error(msg_text)
            elif msg_type == "warning":
                st.warning(msg_text)
            elif msg_type == "info":
                st.info(msg_text)
        # Don't clear messages - let them persist until next PDF processing
    
    # Display debug info if available
    if st.session_state.get('debug_enabled', False) and hasattr(st.session_state, 'extracted_text') and st.session_state.extracted_text:
        with st.expander("🔍 DEBUG: OCR Extracted Text", expanded=False):
            st.write(f"**Extracted text length**: {len(st.session_state.extracted_text)} characters")
            st.text_area("Raw OCR Output", st.session_state.extracted_text, height=200, help="This is the raw text extracted from the PDF using OCR")
    
    if st.session_state.get('debug_enabled', False) and hasattr(st.session_state, 'gpt_response') and st.session_state.gpt_response:
        with st.expander("🤖 DEBUG: GPT Response", expanded=False):
            st.write("**GPT Response:**")
            st.text_area("Full GPT Response", st.session_state.gpt_response, height=300)
    
    # GPT Extracted Bearings Section
    if st.session_state.parsed_bearings:
        st.markdown("# 🧭 Meets and Bounds")
        st.markdown("<small>extracted from Legal Description (please review)</small>", unsafe_allow_html=True)
        
        # Create editable table data with proper structure
        bearing_data = []
        for i, bearing in enumerate(st.session_state.parsed_bearings):
            bearing_data.append({
                "Line": i + 1,
                "Cardinal NS": bearing.get('cardinal_ns', 'North'),
                "Degrees": bearing.get('degrees', 0),
                "Minutes": bearing.get('minutes', 0),
                "Seconds": bearing.get('seconds', 0),
                "Cardinal EW": bearing.get('cardinal_ew', 'East'),
                "Distance": bearing.get('distance', 0.0),
                "Monument": bearing.get('monument', '')
            })
        
        df = pd.DataFrame(bearing_data)
        
        # Make table editable with proper column configuration
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Line": st.column_config.NumberColumn(
                    "Line",
                    disabled=True,
                    width="small"
                ),
                "Cardinal NS": st.column_config.SelectboxColumn(
                    "Cardinal NS",
                    options=["North", "South"],
                    width="small"
                ),
                "Degrees": st.column_config.NumberColumn(
                    "Degrees",
                    min_value=0,
                    max_value=90,
                    step=1,
                    width="small"
                ),
                "Minutes": st.column_config.NumberColumn(
                    "Minutes",
                    min_value=0,
                    max_value=59,
                    step=1,
                    width="small"
                ),
                "Seconds": st.column_config.NumberColumn(
                    "Seconds",
                    min_value=0,
                    max_value=59,
                    step=1,
                    width="small"
                ),
                "Cardinal EW": st.column_config.SelectboxColumn(
                    "Cardinal EW",
                    options=["East", "West"],
                    width="small"
                ),
                "Distance": st.column_config.NumberColumn(
                    "Distance (ft)",
                    min_value=0.0,
                    step=0.01,
                    format="%.2f",
                    width="medium"
                ),
                "Monument": st.column_config.TextColumn(
                    "Monument",
                    width="large"
                )
            },
            key="editable_bearings_table"
        )
        
        # Update session state if table was edited
        if not edited_df.equals(df):
            # Convert edited table back to parsed_bearings format
            updated_bearings = []
            for _, row in edited_df.iterrows():
                bearing = {
                    'cardinal_ns': row['Cardinal NS'],
                    'degrees': int(row['Degrees']),
                    'minutes': int(row['Minutes']),
                    'seconds': int(row['Seconds']),
                    'cardinal_ew': row['Cardinal EW'],
                    'distance': float(row['Distance']),
                    'monument': str(row['Monument']),
                    'original_text': f"{row['Cardinal NS']} {int(row['Degrees'])}° {int(row['Minutes'])}' {int(row['Seconds'])}\" {row['Cardinal EW']}",
                    'bearing': f"{row['Cardinal NS']} {int(row['Degrees'])}° {int(row['Minutes'])}' {int(row['Seconds'])}\" {row['Cardinal EW']}"
                }
                updated_bearings.append(bearing)
            
            st.session_state.parsed_bearings = updated_bearings
            st.success("✅ Table updated! Changes will be used when drawing lines.")
        
        # Action buttons for drawing and exporting
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("🎯 Draw from GPT Data", use_container_width=True, type="primary"):
                st.session_state.current_point = [0, 0]
                st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'bearing_desc', 'distance', 'monument'])
                draw_lines_from_bearings()
                st.success(f"Drew {len(st.session_state.parsed_bearings)} lines from GPT data!")
        
        with col2:
            if st.button("📝 Populate Input Fields", use_container_width=True, type="secondary"):
                # Clear and populate input fields
                st.session_state.line_count = len(st.session_state.parsed_bearings)
                for i, bearing in enumerate(st.session_state.parsed_bearings):
                    st.session_state[f"cardinal_ns_{i}"] = bearing['cardinal_ns']
                    st.session_state[f"degrees_{i}"] = bearing['degrees']
                    st.session_state[f"minutes_{i}"] = bearing['minutes']
                    st.session_state[f"seconds_{i}"] = bearing['seconds']
                    st.session_state[f"cardinal_ew_{i}"] = bearing['cardinal_ew']
                    st.session_state[f"distance_{i}"] = float(bearing['distance'])
                    st.session_state[f"monument_{i}"] = bearing.get('monument', '')
                st.success("Input fields populated! Scroll down to review and edit if needed.")
        
        with col3:
            if st.button("📄 Export DXF", use_container_width=True):
                dxf_data = create_dxf()
                if dxf_data:
                    st.download_button(
                        label="Download DXF",
                        data=dxf_data,
                        file_name="line_drawing.dxf",
                        mime="application/dxf"
                    )
        
        with col4:
            if st.button("📊 Export PDF", use_container_width=True):
                pdf_data = export_pdf()
                if pdf_data:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name="survey_report.pdf",
                        mime="application/pdf"
                    )
        
        st.divider()

    # Line Drawing Section - Collapsible
    with st.expander("⚙️ Manual Input & Advanced Options", expanded=st.session_state.draw_lines_section_expanded):
        st.subheader("Draw Lines")

        # Action Buttons
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            if st.button("Draw Lines", use_container_width=True):
                # Reset current point and lines
                st.session_state.current_point = [0, 0]
                st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'bearing_desc', 'distance', 'monument'])

                # Collect all valid manual bearings
                manual_bearings = []
                for line_num in range(st.session_state.line_count):
                    if st.session_state.get(f"distance_{line_num}", 0) > 0:
                        bearing = manual_bearing_input_to_parsed_format(
                            st.session_state.get(f"cardinal_ns_{line_num}", "North"),
                            st.session_state.get(f"degrees_{line_num}", 0),
                            st.session_state.get(f"minutes_{line_num}", 0),
                            st.session_state.get(f"seconds_{line_num}", 0),
                            st.session_state.get(f"cardinal_ew_{line_num}", "East"),
                            st.session_state.get(f"distance_{line_num}", 0.00),
                            st.session_state.get(f"monument_{line_num}", "")
                        )
                        if bearing:
                            manual_bearings.append(bearing)

                if manual_bearings:
                    st.session_state.parsed_bearings = manual_bearings
                    draw_lines_from_bearings()

        with col2:
            if st.button("Export DXF", use_container_width=True):
                dxf_data = create_dxf()
                if dxf_data:
                    st.download_button(
                        label="Download DXF",
                        data=dxf_data,
                        file_name="line_drawing.dxf",
                        mime="application/dxf"
                    )

        with col3:
            if st.button("Export PDF", use_container_width=True):
                pdf_data = export_pdf()
                if pdf_data:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name="survey_report.pdf",
                        mime="application/pdf"
                    )

        with col4:
            if st.button("Debug", use_container_width=True):
                # Generate 4 random bearings
                for i in range(4):
                    bearing = generate_random_bearing()
                    st.session_state[f"cardinal_ns_{i}"] = bearing['cardinal_ns']
                    st.session_state[f"degrees_{i}"] = bearing['degrees']
                    st.session_state[f"minutes_{i}"] = bearing['minutes']
                    st.session_state[f"seconds_{i}"] = bearing['seconds']
                    st.session_state[f"cardinal_ew_{i}"] = bearing['cardinal_ew']
                    st.session_state[f"distance_{i}"] = bearing['distance']
                    st.session_state[f"monument_{i}"] = bearing['monument']

                # Auto-trigger the draw lines functionality
                st.session_state.current_point = [0, 0]
                st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'bearing_desc', 'distance', 'monument'])

                manual_bearings = []
                for i in range(4):
                    bearing = manual_bearing_input_to_parsed_format(
                        st.session_state[f"cardinal_ns_{i}"],
                        st.session_state[f"degrees_{i}"],
                        st.session_state[f"minutes_{i}"],
                        st.session_state[f"seconds_{i}"],
                        st.session_state[f"cardinal_ew_{i}"],
                        st.session_state[f"distance_{i}"],
                        st.session_state[f"monument_{i}"]
                    )
                    if bearing:
                        manual_bearings.append(bearing)

                if manual_bearings:
                    st.session_state.parsed_bearings = manual_bearings
                    draw_lines_from_bearings()

        with col5:
            if st.button("Clear All", use_container_width=True):
                st.session_state.current_point = [0, 0]
                st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'bearing_desc', 'distance', 'monument'])
                st.session_state.parsed_bearings = None
                st.session_state.extracted_text = None
                st.session_state.pdf_image = None
                st.session_state.supplemental_info = None
                st.session_state.manual_bearing = None
                st.session_state.line_count = 4  # Reset line count

                # Clear all input fields
                for i in range(4):
                    st.session_state[f"cardinal_ns_{i}"] = "North"
                    st.session_state[f"degrees_{i}"] = 0
                    st.session_state[f"minutes_{i}"] = 0
                    st.session_state[f"seconds_{i}"] = 0
                    st.session_state[f"cardinal_ew_{i}"] = "East"
                    st.session_state[f"distance_{i}"] = 0.00
                    st.session_state[f"monument_{i}"] = ""

        with col6:
            if st.button("Add Line", use_container_width=True):
                st.session_state.line_count += 1
                # Initialize new line fields
                i = st.session_state.line_count - 1
                st.session_state[f"cardinal_ns_{i}"] = "North"
                st.session_state[f"degrees_{i}"] = 0
                st.session_state[f"minutes_{i}"] = 0
                st.session_state[f"seconds_{i}"] = 0
                st.session_state[f"cardinal_ew_{i}"] = "East"
                st.session_state[f"distance_{i}"] = 0.00
                st.session_state[f"monument_{i}"] = ""

        # Create a container for all line inputs
        with st.container():
            for line_num in range(st.session_state.line_count):
                st.markdown(f"**Line {line_num + 1}**")
                col1, col2, col3, col4, col5, col6, col7 = st.columns([2,1,1,1,1.5,1.5,2])

                # Use the value from session_state if it exists, otherwise use a default
                with col1:
                    st.selectbox("Cardinal", ("North", "South"), key=f"cardinal_ns_{line_num}")
                with col2:
                    st.number_input("Degrees", min_value=0, max_value=90, key=f"degrees_{line_num}")
                with col3:
                    st.number_input("Minutes", min_value=0, max_value=59, key=f"minutes_{line_num}")
                with col4:
                    st.number_input("Seconds", min_value=0, max_value=59, key=f"seconds_{line_num}")
                with col5:
                    st.selectbox("Cardinal", ("East", "West"), key=f"cardinal_ew_{line_num}")
                with col6:
                    st.number_input("Distance (feet)", min_value=0.0, key=f"distance_{line_num}")
                with col7:
                    st.text_input("Monument", key=f"monument_{line_num}")
                st.divider()

    # Display the plot
    fig = draw_lines()
    st.plotly_chart(fig)

    # Display supplemental information if available
    if st.session_state.supplemental_info:
        st.subheader("Property Information")
        col1, col2, col3 = st.columns([4, 3, 3])

        with col1:
            st.metric("Land Lot", st.session_state.supplemental_info.get('land_lot', 'N/A'))
        with col2:
            st.metric("District", st.session_state.supplemental_info.get('district', 'N/A'))
        with col3:
            st.metric("County", st.session_state.supplemental_info.get('county', 'N/A'))

    # Display PDF image if available
    if st.session_state.pdf_image:
        st.subheader("PDF Document")
        st.write("Please review your document shown below to verify the system correctly recognized the meets and bounds")
        st.image(st.session_state.pdf_image, caption="PDF First Page", use_container_width=True)

if __name__ == "__main__":
    main()