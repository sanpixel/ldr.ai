# LDR Application Documentation

## Warp Usage Information
Warp usage: 781
Current session: Clean repository setup with all commit history removed for security
Application: Legal Description Reader - AI-powered property boundary mapping tool

---

## 1. Overview

The LDR (Line Drawing a.k.a Legal Description Reader) application is a web-based tool built with Streamlit that automates the process of converting property legal descriptions from PDF documents into digital line drawings. It is designed to assist surveyors, real estate professionals, and legal teams by providing a quick and accurate way to visualize property boundaries.

The application leverages Optical Character Recognition (OCR) and Natural Language Processing (NLP) to extract bearing, distance, and monument information from unstructured text. The extracted data is then used to generate a 2D plot of the property lines, which can be exported to DXF, and PDF formats.

## 2. Core Features

### 2.1. PDF Upload and Processing
- **File Upload**: Users can upload PDF documents containing legal descriptions. The application includes security checks to ensure only PDF files are accepted and provides recommendations for handling file size and validation in production environments.
- **OCR Text Extraction**: It uses `pytesseract` and `pdf2image` to convert PDF pages into images and then extracts the text. This allows it to process both text-based and scanned PDFs.
- **Multi-Page Support**: The application can process multi-page documents by concatenating the text from all pages.

### 2.2. Intelligent Data Extraction with GPT
- **Bearing and Distance Parsing**: The application uses a fine-tuned GPT-3.5 model from OpenAI to parse the extracted text. This model is trained to recognize a wide variety of bearing formats, including:
    - Standard surveyor's notation (e.g., "North 45 degrees 30 minutes East").
    - Decimal bearings (e.g., "N 45.5 E").
    - Abbreviated formats (e.g., "N45°30'E").
- **Monument and Reference Recognition**: The GPT model also identifies and extracts references to monuments (e.g., "to an iron pin," "along a creek") and adjacent properties, which are critical for accurate boundary mapping.
- **Flexible and Robust**: The use of a fine-tuned LLM makes the extraction process highly flexible and resilient to variations in legal description language and formatting. This is a significant improvement over traditional regex-based methods, which are brittle and difficult to maintain.

### 2.3. Interactive Line Drawing
- **2D Visualization**: The application generates an interactive 2D plot of the property boundaries using `plotly`. Users can pan, zoom, and inspect the drawing to verify its accuracy.
- **Point of Beginning (POB)**: The plot clearly marks the Point of Beginning (POB) and annotates each line segment with its bearing and distance.
- **Dynamic Updates**: The plot is automatically updated whenever the underlying data changes, providing immediate feedback to the user.

### 2.4. Manual Data Entry and Correction
- **Manual Input**: In cases where the automated extraction is incomplete or inaccurate, users can manually enter or correct the bearing, distance, and monument information for each line segment.
- **Dynamic Line Management**: Users can add or remove lines as needed, and the application will dynamically adjust the input fields and the plot.
- **Random Bearing Generation**: A "Debug" feature allows developers to quickly generate random, realistic bearings for testing and demonstration purposes.

### 2.5. Data Export
- **DXF Export**: The application can export the line drawing to a DXF (Drawing Exchange Format) file, which is compatible with most CAD software. The export includes lines, points, and annotations, making it easy to import into other systems for further analysis or drafting.
- **PDF Export**: Users can also export a comprehensive survey report in PDF format. This report includes:
    - A summary of the property information (Land Lot, District, County).
    - A disclaimer regarding the accuracy of computer-recognized bearings.
    - The 2D line drawing.
    - A detailed table of all bearings, distances, and monuments.

## 3. Technical Architecture

### 3.1. Frontend
- **Framework**: The user interface is built with [Streamlit](https://streamlit.io/), a Python library for creating data-centric web applications.
- **Plotting**: Interactive plots are generated using [Plotly](https://plotly.com/), which provides a rich set of tools for creating and customizing charts.

### 3.2. Backend
- **Language**: The entire application is written in Python 3.11.
- **Data Extraction**:
    - **OCR**: `pytesseract` and `pdf2image` are used for text extraction from PDFs.
    - **NLP**: A fine-tuned `gpt-3.5-turbo-0125` model from OpenAI is used for intelligent parsing of legal descriptions.
- **CAD/DXF Handling**: The `ezdxf` library is used to create and manipulate DXF files. The application also includes optional support for FreeCAD, although it is not a required dependency.
- **PDF Generation**: The `reportlab` library is used to generate the PDF survey reports.
- **Dependencies**: All required packages are listed in `requirements.txt` and `pyproject.toml`.

### 3.3. Configuration and Deployment
- **Environment Variables**: The application uses a `.env` file to manage sensitive information, such as the OpenAI API key.
- **CORS**: Cross-Origin Resource Sharing (CORS) is enabled in `.streamlit/config.toml` to allow file uploads from different origins. This is necessary for the PDF upload feature to work correctly in all browsers.
- **Deployment**: The application is configured to run on port 5000 and is ready for deployment. The `README.md` file provides a security checklist and configuration details for production environments.

## 4. How to Use the Application

1. **Upload a PDF**: Click the "Choose a PDF file" button to upload a document containing a legal description.
2. **Process the PDF**: Click the "Process PDF" button to start the OCR and data extraction process. The application will display the extracted text and a preview of the first page of the PDF.
3. **Review and Correct**: The extracted bearings, distances, and monuments will be populated in the input fields below the plot. Review this information for accuracy and make any necessary corrections.
4. **Draw the Lines**: Click the "Draw Lines" button to generate the 2D plot based on the data in the input fields.
5. **Export the Drawing**:
    - Click "Export DXF" to download the drawing as a DXF file.
    - Click "Export PDF" to download a complete survey report in PDF format.

## 5. Key Functions and Components

### 5.1. Main Functions
- `extract_bearings_with_gpt()`: Uses OpenAI's GPT model to parse legal descriptions and extract bearings, distances, and monuments
- `process_pdf()`: Handles PDF file upload, OCR processing, and text extraction
- `draw_lines_from_bearings()`: Creates the 2D visualization from extracted bearing data
- `create_dxf()`: Generates DXF files for CAD software compatibility
- `export_pdf()`: Creates comprehensive PDF survey reports

### 5.2. Data Processing
- **Coordinate System**: Uses Georgia State Plane Coordinate System as reference
- **Bearing Conversion**: Converts between DMS (Degrees, Minutes, Seconds) and decimal degrees
- **Line Calculation**: Uses trigonometry to calculate endpoint coordinates from bearings and distances

### 5.3. Session State Management
- Maintains user data across Streamlit sessions
- Stores extracted text, parsed bearings, PDF images, and property information
- Enables dynamic line management and real-time updates

## 6. File Structure

```
ldr/
├── main.py                              # Main application file
├── requirements.txt                     # Python dependencies
├── pyproject.toml                       # Project configuration
├── README.md                           # Deployment and security guide
├── .env                                # Environment variables (API keys)
├── .streamlit/config.toml              # Streamlit configuration
├── classified_legal_descriptions.jsonl # Training data for GPT model
├── fixed_classified_legal_descriptions.jsonl
├── ocr_legal_description.jsonl
└── .cursor/rules/                      # Development guidelines
    ├── core.md
    ├── refresh.md
    └── request.md
```

## 7. Dependencies

### Core Libraries
- `streamlit`: Web application framework
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `plotly`: Interactive plotting
- `ezdxf`: DXF file creation
- `pytesseract`: OCR text extraction
- `pdf2image`: PDF to image conversion
- `openai`: GPT API integration
- `reportlab`: PDF generation

### Optional Dependencies
- `FreeCAD`: Advanced CAD functionality (not required)

## 8. Security Considerations

- File upload validation (PDF files only)
- CORS configuration for cross-origin requests
- API key management through environment variables
- Input sanitization and error handling
- Production deployment security checklist included in README.md

## 9. Future Enhancement Opportunities

- Support for additional file formats (TIFF, DOC, etc.)
- Integration with GIS systems
- Batch processing capabilities
- Advanced error correction and validation
- Multi-language support for legal descriptions
- Cloud-based deployment options
