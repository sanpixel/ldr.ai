from dotenv import load_dotenv
load_dotenv()

import os
print("API Key exists:", bool(os.environ.get("OPENAI_API_KEY")))

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
from openai import OpenAI
import json
import io
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

def extract_bearings_with_gpt(text):
    """Use GPT to extract bearings from text with a robust, line-by-line parser."""
    try:
        prompt = """Extract all bearings, distances, and monuments from the following legal description text. 
        For each line segment found, output in this exact format:
        BEARING: [bearing]
        DISTANCE: [distance]
        MONUMENT: [monument/reference]

        Ensure each attribute is on a new line.
        Text to analyze:
        """ + text

        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:personal:ldr:BEoe3v67",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        result_text = response.choices[0].message.content
        bearings = []
        current_bearing = {}
        lines = result_text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith('BEARING:'):
                if current_bearing.get('distance'):
                    bearings.append(current_bearing)
                
                bearing_text = line.split(':', 1)[1].strip()
                current_bearing = {'bearing': bearing_text} 

                pattern = r'(S|South|N|North)[\s\.]*(\d+)[\s°degr\']*(?:(\d+)[\s\'min]*)?(?:(\d*\.?\d*)[\s"sec]*)?[\s\.]*(E|W|East|West)'
                match = re.search(pattern, bearing_text, re.IGNORECASE)

                if match:
                    st.write(f"**DEBUG: Successfully matched bearing text: '{bearing_text}'**")
                    groups = match.groups()
                    
                    ns_raw = (groups[0] or '').upper()
                    ew_raw = (groups[4] or '').upper()

                    current_bearing['cardinal_ns'] = 'South' if 'S' in ns_raw else 'North'
                    current_bearing['cardinal_ew'] = 'East' if 'E' in ew_raw else 'West'
                    current_bearing['degrees'] = int(groups[1]) if groups[1] else 0
                    current_bearing['minutes'] = int(groups[2]) if groups[2] else 0
                    current_bearing['seconds'] = int(float(groups[3])) if groups[3] else 0
                    current_bearing['original_text'] = bearing_text  # Store the original bearing text
                else:
                    st.write(f"**DEBUG: FAILED to match bearing text: '{bearing_text}'**")
                    current_bearing['original_text'] = bearing_text  # Store even if parsing failed

            elif line.upper().startswith('DISTANCE:') and current_bearing:
                distance_text = line.split(':', 1)[1].strip()
                distance_match = re.search(r'(\d+(?:\.\d+)?)', distance_text)
                if distance_match:
                    current_bearing['distance'] = float(distance_match.group(1))

            elif line.upper().startswith('MONUMENT:') and current_bearing:
                current_bearing['monument'] = line.split(':', 1)[1].strip()

        if current_bearing.get('distance'):
            bearings.append(current_bearing)

        return [b for b in bearings if 'cardinal_ns' in b], result_text # Return only fully parsed bearings

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
        st.session_state.draw_lines_section_expanded = True  # Start expanded by default
    
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

        # Extract supplemental information first
        if os.environ.get("OPENAI_API_KEY"):
            st.info("Extracting property information...")
            try:
                supplemental_info = extract_supplemental_info_with_gpt(extracted_text)
                if supplemental_info:
                    st.session_state.supplemental_info = supplemental_info
                    st.success("Successfully extracted property information")
            except Exception as e:
                st.error(f"Error extracting property information: {str(e)}")

        # First try GPT extraction for bearings
        if os.environ.get("OPENAI_API_KEY"):
            st.info("Using GPT to analyze the text for bearings...")
            try:
                bearings, result_text = extract_bearings_with_gpt(extracted_text)
                st.write(f"**DEBUG: GPT returned {len(bearings)} bearings**")
                if bearings:
                    return bearings
                else:
                    st.warning("No bearings found")
                    st.write("**DEBUG: GPT Response:**")
                    st.write(result_text)
                    return []
            except Exception as e:
                st.error(f"GPT analysis failed: {str(e)}")
                st.write(f"GPT Error Details: {str(e)}")
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

def main():
    st.set_page_config(layout="wide")
    st.title("Line Drawing Application")
    initialize_session_state()

    # Create two columns for the main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # PDF Upload Section
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        if uploaded_file is not None:
            if st.button("Process PDF"):
                with st.spinner('Processing PDF...'):
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
                        st.rerun()

    with col2:
        if st.session_state.pdf_image:
            st.image(st.session_state.pdf_image, caption='PDF Preview', width=250)

    # GPT Extracted Bearings Section
    if st.session_state.parsed_bearings:
        st.subheader("🧭 Meets and Bounds extracted from Legal Description (please review)")
        
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
        
        # Custom CSS for all buttons
        st.markdown("""
        <style>
        /* Green primary button for Draw from GPT Data */
        .stButton > button[kind="primary"] {
            background-color: #28a745 !important;
            border-color: #28a745 !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #218838 !important;
            border-color: #1e7e34 !important;
        }
        
        /* Light blue secondary button for Populate Input Fields */
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
        
        with col1:
            if st.button("🎯 Draw from GPT Data", use_container_width=True, type="primary"):
                st.session_state.current_point = [0, 0]
                st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'bearing_desc', 'distance', 'monument'])
                draw_lines_from_bearings()
                st.success(f"Drew {len(st.session_state.parsed_bearings)} lines from GPT data!")
                st.rerun()
        
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
                st.rerun()
        
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