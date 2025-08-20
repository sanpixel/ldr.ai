# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Legal Description Reader (LDR.ai) is a Streamlit web application that converts property legal descriptions from PDF documents into interactive line drawings and CAD files. The application uses AI (OpenAI GPT) to parse complex legal descriptions and extract bearing, distance, and monument information.

**Core Technology Stack:**
- **Framework**: Streamlit (Python web app framework)
- **AI Processing**: OpenAI GPT-3.5-turbo (fine-tuned model for legal descriptions)
- **OCR**: Tesseract + pytesseract for PDF text extraction
- **Plotting**: Plotly for interactive 2D visualizations
- **CAD Export**: ezdxf for DXF file generation
- **PDF Generation**: ReportLab for survey reports
- **Deployment**: Google Cloud Run via Docker

## Development Commands

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (development)
streamlit run main.py

# Run locally on specific port/host (matches production config)
streamlit run main.py --server.port=5000 --server.address=0.0.0.0
```

### Environment Setup
```bash
# Copy and configure environment variables
cp .env.example .env
# Edit .env with your OpenAI API key and other credentials

# Required environment variables:
# OPENAI_API_KEY - Required for GPT-based legal description parsing
# SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY - For authentication (planned)
# GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET - For OAuth (planned)
```

### Testing & Development
```bash
# Test with sample PDFs (debug mode)
# The app includes test buttons for built-in PDFs:
# - combine_SNAPFINGER_TRACT-1_LD.pdf
# - combine_SNAPFINGER_TRACT-2_LD.pdf  
# - combine_SNAPFINGER_TRACT-3_LD.pdf

# Generate random test data using the "Debug" button in the UI
```

### Building & Deployment
```bash
# Build Docker image
docker build -t ldr-ai .

# Run Docker container locally
docker run -p 8080:8080 ldr-ai

# Deploy via Google Cloud Build (configured in cloudbuild.yaml)
gcloud builds submit --config cloudbuild.yaml

# Deploy via GitHub Actions (recommended - see TODO.md)
# Uses workflow files in .github/workflows/
```

## Architecture Overview

### Core Processing Pipeline
1. **PDF Upload** → `process_pdf()` → OCR text extraction
2. **AI Analysis** → `extract_bearings_with_gpt()` → Parse legal descriptions
3. **Data Validation** → Manual input forms → Bearing/distance verification
4. **Visualization** → `draw_lines_from_bearings()` → Interactive plot generation
5. **Export** → `create_dxf()`, `export_pdf()` → CAD and report generation

### Key Functions & Components

**PDF Processing Functions:**
- `process_pdf()` - Main PDF upload and OCR processing
- `extract_bearings_with_gpt()` - AI-powered legal description parsing
- `extract_supplemental_info_with_gpt()` - Property information extraction

**Data Conversion Functions:**
- `dms_to_decimal()` - Convert surveyor bearings to decimal degrees
- `decimal_to_dms()` - Convert back to degrees/minutes/seconds format
- `calculate_endpoint()` - Calculate line endpoints from bearing/distance

**Visualization Functions:**
- `draw_lines()` - Create interactive Plotly visualization
- `draw_lines_from_bearings()` - Process bearing data into line segments

**Export Functions:**
- `create_dxf()` - Generate DXF files for CAD software
- `export_pdf()` - Create comprehensive survey reports
- `export_cad()` - FreeCAD integration (optional)

### Session State Management
The app heavily uses Streamlit session state to maintain data across interactions:
- `st.session_state.parsed_bearings` - Extracted bearing data
- `st.session_state.lines` - Generated line segments (DataFrame)
- `st.session_state.current_point` - Drawing cursor position
- `st.session_state.extracted_text` - OCR results from PDF
- `st.session_state.supplemental_info` - Property details (Land Lot, District, County)

### Fine-tuned GPT Model
The application uses a custom fine-tuned GPT model (`ft:gpt-3.5-turbo-0125:personal:ldr:BEoe3v67`) specifically trained on legal descriptions. Training data is stored in `classified_legal_descriptions.jsonl`.

## Configuration Files

### Streamlit Configuration (`.streamlit/config.toml`)
- CORS enabled for file uploads
- Configured for deployment on 0.0.0.0:5000
- XSRF protection disabled for development

### Docker Configuration
- Based on Python 3.11-slim
- Includes Tesseract OCR and Poppler utilities
- Configured for Cloud Run deployment on port 8080

### Dependencies
- **Primary**: `requirements.txt` (runtime dependencies)
- **Development**: `pyproject.toml` (project metadata and version pins)

## Working with Legal Descriptions

### Supported Formats
The AI model can parse various legal description formats:
- **Metes and Bounds**: "North 45 degrees 30 minutes East 150.0 feet"
- **Abbreviated**: "N 45° 30' E 150.0'"
- **Decimal**: "N 45.5° E 150.0'"
- **Verbose**: "North forty-five degrees thirty minutes East"

### Coordinate System
- Uses Georgia State Plane Coordinate System as reference
- Point of Beginning (POB) at origin (0,0)
- Supports monument and reference point tracking

### Quality Assurance
- Debug mode enabled by default (`DEBUG_MODE = True`)
- Extensive logging of GPT parsing results
- Manual correction interface for AI-extracted data
- Closure error calculation (planned - see TODO.md)

## Development Guidelines

### Error Handling
- Comprehensive try/catch blocks around OCR and AI operations
- Graceful degradation when OpenAI API is unavailable
- File validation for PDF uploads only
- Session state error recovery

### Performance Considerations
- OCR processing can be memory-intensive for large PDFs
- GPT API calls have rate limiting considerations
- Plotly visualizations are optimized for typical property sizes
- Export operations use streaming for large datasets

### Code Patterns
Follow the established patterns in the codebase:
- Streamlit session state for data persistence
- DataFrame operations for line data management
- Regex patterns for bearing text parsing (fallback to GPT)
- Error messages displayed via `st.error()` and `st.warning()`

## Project Planning & Next Steps

### Check Current Priorities
Always review these files before starting work:
- **PRD.md** - Complete project requirements and technical specifications
- **TODO.md** - Current task list with priorities and implementation notes

### Current High Priority Items (from TODO.md)

**High Priority:**
- Closure line calculations and error reporting
- Improved legal description type detection (plat reference, government survey)
- User authentication with Google OAuth + Supabase
- GitHub Actions deployment pipeline

**UI/UX Improvements:**
- React frontend migration (keeping Python backend)
- Mobile responsiveness improvements
- Better visual feedback during processing

**Technical Features:**
- Support for curves and arcs
- Area calculations for closed polygons
- Additional CAD export formats
- Coordinate system transformations

## Security Notes

- API keys managed via environment variables
- File upload restricted to PDF only
- CORS configuration required for cross-origin uploads
- Production deployment security checklist in README.md
- OAuth implementation planned for user authentication

## Dependencies & System Requirements

**System Dependencies:**
- Tesseract OCR (`tesseract-ocr`, `tesseract-ocr-eng`)
- Poppler utilities (`poppler-utils`)
- Python 3.11+

**Python Dependencies:**
- Core: streamlit, openai, pytesseract, pdf2image
- Math/Data: numpy, pandas, plotly
- CAD/Export: ezdxf, reportlab
- Optional: FreeCAD (advanced CAD features)

**Development Tools:**
- Cursor AI rules configured in `.cursor/rules/`
- Conventional commit standards enforced
- Path validation protocols for file operations
