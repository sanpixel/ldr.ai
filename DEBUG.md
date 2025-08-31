# DEBUG.md - Legal Description Reader Debug Information

## Overview
This document contains comprehensive debug information, configuration options, and troubleshooting details for the Legal Description Reader application.

## Debug Configuration

### Environment Variables
```python
DEBUG_MODE = True                    # Enables debug output throughout the app
AUTO_PROCESS_DEBUG = True           # Auto-processes first local PDF for testing
```

### Debug Features Enabled
When `DEBUG_MODE = True`:
- OCR extracted text display in expandable section
- GPT response display in expandable section
- Classification reasoning display with confidence metrics
- Database save status notifications
- File processing status messages
- Bearing parsing match details
- Session state key ordering information

## Auto-Processing Debug Mode

### Configuration
```python
AUTO_PROCESS_DEBUG = True  # Auto-process first PDF for testing data collection
```

### Behavior
- **Trigger**: Runs automatically on first app load when enabled
- **Target**: First available local PDF file in project directory
- **Prevention**: Uses `st.session_state.auto_processed` flag to prevent repeat processing
- **Process**: Identical to manual "Process PDF" button functionality

### Implementation Details
```python
if AUTO_PROCESS_DEBUG and 'auto_processed' not in st.session_state:
    st.session_state.auto_processed = True
    first_pdf = pdf_files[0]
    
    # Creates BytesIO buffer with filename attribute
    pdf_buffer = BytesIO(file_content)
    pdf_buffer.name = first_pdf
    
    # Calls same process_pdf() function as manual processing
    bearings = process_pdf(pdf_buffer)
```

### Status Messages
- `🤖 AUTO-PROCESSING: {filename} for data collection testing...`
- `✅ AUTO-PROCESSED: Extracted {count} bearings from {filename}!`
- `⚠️ AUTO-PROCESS: No bearings found in {filename}`
- `❌ AUTO-PROCESS ERROR: {error_message}`

## Database Integration Debug

### Supabase Connection
- **Database**: PostgreSQL via Supabase
- **Table**: `classification_data`
- **Debug Output**: Save status notifications when `DEBUG_MODE = True`

### Generated Columns (21 fields)
```sql
-- Processing metrics
processing_time FLOAT GENERATED ALWAYS AS ((reasoning_data->>'processing_time')::FLOAT) STORED,
bearing_count INTEGER GENERATED ALWAYS AS ((reasoning_data->>'bearing_count')::INTEGER) STORED,

-- File metadata
file_size BIGINT GENERATED ALWAYS AS ((reasoning_data->>'file_size')::BIGINT) STORED,
file_hash TEXT GENERATED ALWAYS AS (reasoning_data->>'file_hash') STORED,

-- Quality metrics
parsing_success_rate FLOAT GENERATED ALWAYS AS ((reasoning_data->>'parsing_success_rate')::FLOAT) STORED,
ocr_confidence FLOAT GENERATED ALWAYS AS ((reasoning_data->>'ocr_confidence')::FLOAT) STORED,
model_confidence FLOAT GENERATED ALWAYS AS ((reasoning_data->>'model_confidence')::FLOAT) STORED,

-- User/session context
upload_method TEXT GENERATED ALWAYS AS (reasoning_data->>'upload_method') STORED,
debug_mode BOOLEAN GENERATED ALWAYS AS ((reasoning_data->>'debug_mode')::BOOLEAN) STORED,
session_id TEXT GENERATED ALWAYS AS (reasoning_data->>'session_id') STORED,

-- Processing context
gpt_model TEXT GENERATED ALWAYS AS (reasoning_data->>'gpt_model') STORED,
gpt_temperature FLOAT GENERATED ALWAYS AS ((reasoning_data->>'gpt_temperature')::FLOAT) STORED
```

### Indexing Strategy
All 21 generated columns have individual B-tree indexes for analytics queries:
```sql
CREATE INDEX idx_classification_data_filename ON classification_data (filename);
CREATE INDEX idx_classification_data_user_email ON classification_data (user_email);
-- ... (19 more indexes for generated columns)
```

## GPT Integration Debug

### Model Configuration
```python
model="ft:gpt-3.5-turbo-0125:personal:ldr:BEoe3v67"  # Fine-tuned model
temperature=0.1  # Low temperature for consistent results
```

### Classification Types
1. **EXPLICIT_BEARINGS**: Contains specific measurements (degrees, minutes, seconds, distances)
2. **ABSTRACT_BEARINGS**: Directional descriptions without specific measurements
3. **EXTERNAL_REF**: References to external documents or survey systems

### Debug Output Structure
```json
{
  "filename": "document.pdf",
  "user_email": "user@example.com",
  "timestamp": "2025-08-31T20:30:17.123456",
  "input_text": "First 1000 chars of extracted text...",
  "full_response": "Complete GPT response...",
  "classification": "explicit_bearings|abstract_bearings|external_ref",
  "confidence": "high|medium|low",
  "reasoning": "Explanation of classification decision",
  "evidence": "Specific text supporting the decision",
  "alternatives": "Second and third choice classifications with reasons"
}
```

### Bearing Parsing Patterns
```regex
# Standard pattern: S 73° 32' 01" W
pattern = r'(S|South|N|North)[\s\.]*(\\d+)(?:[\s°degrees]+(?:(\\d+)(?:[\s\'minutes]+(?:(\\d+(?:\\.\\d+)?)(?:[\s"seconds]+)?)?)?)?)?[\s]*(E|W|East|West)'

# Long format: North 71 degrees 53 minutes 10 seconds East
long_pattern = r'(North|South)\\s+(\\d+)\\s+degrees?\\s+(\\d+)\\s+minutes?\\s+(\\d+(?:\\.\\d+)?)\\s+seconds?\\s+(East|West)'
```

## Session State Management

### Key Variables
```python
# Core data
'lines'                    # DataFrame with line geometry
'current_point'           # [x, y] coordinates for next line start
'parsed_bearings'         # List of extracted bearing objects
'extracted_text'          # Raw OCR text from PDF
'pdf_image'              # First page image for display
'supplemental_info'      # Land lot, district, county data

# Processing state
'gpt_response'           # Full GPT response for debug
'processing_messages'    # Status messages from PDF processing
'auto_processed'         # Flag to prevent repeat auto-processing

# UI state
'line_count'                    # Number of manual input lines
'draw_lines_section_expanded'   # Expander state for manual input

# Input fields (0-19 for up to 20 lines)
'cardinal_ns_{i}'        # North/South selection
'degrees_{i}'            # Degrees value
'minutes_{i}'            # Minutes value  
'seconds_{i}'            # Seconds value
'cardinal_ew_{i}'        # East/West selection
'distance_{i}'           # Distance in feet
'monument_{i}'           # Monument description
```

### Initialization
```python
def initialize_session_state():
    # Creates empty DataFrames with explicit dtypes
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
    
    # Initializes 20 sets of input fields
    for i in range(20):
        st.session_state[f'cardinal_ns_{i}'] = "North"
        # ... (other field initializations)
```

## File Processing Pipeline

### PDF Processing Flow
1. **File Input**: Upload or local file selection
2. **OCR Extraction**: pdf2image + pytesseract
3. **GPT Analysis**: Classification and bearing extraction
4. **Data Storage**: Save to Supabase database
5. **UI Update**: Populate session state and display results

### Supported File Sources
- **Direct Upload**: Streamlit file uploader
- **Local Files**: PDF files in project directory
- **Google Drive**: Public folder integration (requires API key)

### Error Handling
```python
try:
    # Processing logic
    pass
except Exception as e:
    if DEBUG_MODE:
        st.error(f"Detailed error: {str(e)}")
    else:
        st.error("Processing failed. Please try again.")
```

## Export Capabilities

### DXF Export
- **Library**: ezdxf
- **Features**: Lines, dimensions, monuments, POB annotation
- **Coordinate System**: 2D Cartesian with origin at POB

### PDF Export
- **Library**: ReportLab
- **Content**: Property info, line drawing, bearing table
- **Layout**: Professional survey report format

### Coordinate Calculations
```python
def calculate_endpoint(start_point, bearing, distance):
    bearing_rad = radians(bearing)
    dx = distance * np.sin(bearing_rad)
    dy = distance * np.cos(bearing_rad)
    end_x = start_point[0] + dx
    end_y = start_point[1] + dy
    return [end_x, end_y]

def dms_to_decimal(degrees, minutes, seconds, cardinal_ns, cardinal_ew):
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
```

## Authentication System

### Google OAuth Integration
```python
# Supabase Auth with Google provider
supabase.auth.sign_in_with_oauth({
    "provider": "google",
    "options": {
        "redirect_to": f"{base_url}/?page=auth"
    }
})
```

### Session Management
- **Storage**: Browser localStorage via st-localstorage-connection
- **Tokens**: Access and refresh tokens stored securely
- **User Data**: Email, name, profile picture from Google

## UI/UX Features

### Responsive Design
- **Layout**: Wide layout with sidebar navigation
- **Columns**: Flexible column layouts for different screen sizes
- **Containers**: Proper container usage for content organization

### Interactive Elements
- **Data Editor**: Editable bearing table with column validation
- **Expandable Sections**: Collapsible debug information
- **Progress Indicators**: Spinners for processing operations
- **Status Messages**: Color-coded success/warning/error messages

### Button Styling
```css
/* Green primary buttons for main actions */
.stButton > button[kind="primary"] {
    background-color: #28a745 !important;
    border-color: #28a745 !important;
}

/* Light blue secondary buttons for utility actions */
.stButton > button[kind="secondary"] {
    background-color: #17a2b8 !important;
    border-color: #17a2b8 !important;
    color: white !important;
}
```

## Performance Considerations

### File Size Limits
- **PDF Processing**: No hard limit, but larger files take longer
- **Database Storage**: 1GB limit for file attachments in Supabase
- **Memory Usage**: BytesIO buffers for file processing

### Optimization Strategies
- **Deduplication**: File hashing to prevent duplicate storage
- **Caching**: Session state preservation across reruns
- **Lazy Loading**: Expandable sections for debug information

## Troubleshooting Guide

### Common Issues

#### 1. Auto-Processing Not Working
- **Check**: `AUTO_PROCESS_DEBUG = True` is set
- **Check**: PDF files exist in project directory
- **Check**: Session state flag hasn't been set: `st.session_state.auto_processed`
- **Solution**: Clear browser cache or restart Streamlit

#### 2. Database Connection Failures
- **Check**: Supabase environment variables are set
- **Check**: Internet connectivity
- **Check**: Database schema matches expected structure
- **Solution**: Verify credentials in `.env` file

#### 3. GPT API Errors
- **Check**: OpenAI API key is valid and has credits
- **Check**: Model ID is correct: `ft:gpt-3.5-turbo-0125:personal:ldr:BEoe3v67`
- **Check**: Request rate limits
- **Solution**: Implement retry logic with exponential backoff

#### 4. OCR Quality Issues
- **Check**: PDF quality and resolution
- **Check**: Tesseract installation and PATH
- **Check**: pdf2image dependencies (poppler)
- **Solution**: Preprocess images or use alternative OCR

#### 5. Bearing Parsing Failures
- **Symptom**: GPT classifies as "explicit_bearings" but no bearings extracted
- **Debug**: Check regex patterns match actual text format
- **Debug**: Review GPT response in debug section
- **Solution**: Update parsing patterns or improve prompt

### Debug Commands

#### Session State Inspection
```python
if DEBUG_MODE:
    st.write("Session State Keys:", list(st.session_state.keys()))
    st.write("Parsed Bearings:", st.session_state.parsed_bearings)
    st.write("Current Point:", st.session_state.current_point)
```

#### Database Query Testing
```python
# Test database connection
from utils.classification import save_classification_data
test_data = {"filename": "test.pdf", "user_email": "test@example.com"}
result = save_classification_data(test_data)
st.write("Database test result:", result)
```

#### File Processing Testing
```python
# Test with minimal PDF
if st.button("Test File Processing"):
    with open("test.pdf", "rb") as f:
        content = f.read()
    buffer = BytesIO(content)
    buffer.name = "test.pdf"
    result = process_pdf(buffer)
    st.write("Processing result:", result)
```

## Development Workflow

### Branch Strategy
- **dev**: Development branch for new features
- **prod**: Production branch for stable releases
- **Latest commit**: `ac0602f` (at start of metric enhancement work)

### Testing Strategy
1. **Unit Testing**: Individual function validation
2. **Integration Testing**: Full PDF processing pipeline
3. **User Testing**: Manual workflow verification
4. **Performance Testing**: Large file processing

### Deployment Considerations
- **Environment Variables**: Secure credential management
- **File Dependencies**: Ensure all required files are included
- **Database Migrations**: Handle schema updates gracefully
- **Backup Strategy**: Regular database backups

## Future Enhancements

### Planned Metrics Collection
- **Processing Time**: Time taken for each processing stage
- **File Metadata**: Size, hash, type information
- **Quality Metrics**: OCR confidence, parsing success rates
- **User Analytics**: Usage patterns, error rates
- **Model Performance**: GPT response quality metrics

### Infrastructure Improvements
- **Caching**: Redis or similar for session management
- **Queue System**: Background processing for large files
- **Monitoring**: Application performance monitoring
- **Scaling**: Container orchestration for high load

---

## Contact & Support

For issues or questions related to debug features:
1. Check this documentation first
2. Review session state and error messages
3. Enable `DEBUG_MODE = True` for detailed output
4. Check database and API connectivity
5. Consult GPT response analysis for classification issues

Last Updated: 2025-08-31
Version: 1.0.0
Environment: Windows PowerShell 5.1.22621.5697
