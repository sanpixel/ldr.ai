# Legal Description Reader - Backlog

All tasks, bugs, and ideas consolidated in one place.

## 1. 🎯 High Priority

### 1.1 Fix filename handling and login button formatting
- [x] 1.1.1 Ensure filename field is always preserved in reasoning_data before database save
- [ ] 1.1.2 Put filename first in reasoning_data JSON object for better readability
- [x] 1.1.3 Update login button to be compact without title for PDF preview section

### 1.2 Update classification handler
- [ ] 1.2.1 External ref would usually but not always be classified when 0 bearings are present or a specific plat book is given Example: "Lot 9, Block D of Glenmar II subdivision" alone is NOT external_ref unless it has plat book/page reference and/or zero bearings
- [ ] 1.2.2 Abstract bearings aren't being caught

### 1.3 Fix DB upload_method to distinguish photo uploads
- [ ] 1.3.1 Currently shows "file_upload" for both PDFs and photos
- [ ] 1.3.2 Need to show "photo_upload" when image files (.jpg, .jpeg) are used
- [ ] 1.3.3 Keep "file_upload" for PDFs, "local_file" for local, "google_drive" for Drive

### 1.4 Email PDF reports
- [ ] 1.4.1 Email functionality

### 1.5 Review /pages/prompts usage
- [ ] 1.5.1 Not using /pages/prompts properly, maybe there's a better way

## 2. 🔄 Improvements

### 2.1 Handle abstract bearings in parser
- [ ] 2.1.1 GPT sometimes returns "Northwesterly" instead of "N 45d 30m 15s W"
- [ ] 2.1.2 Parser skips these, causing inconsistent bearing counts
- [ ] 2.1.3 Convert abstract bearings (Northwesterly, Southeasterly, etc.) to approximate degrees
- [ ] 2.1.4 Example: Northwesterly = N 45° 0' 0" W

### 2.2 Improve error handling for malformed PDFs
- [ ] 2.2.1 Better error messages and recovery

### 2.3 Add support for more legal description formats
- [ ] 2.3.1 Expand format support

### 2.4 Handle Point of Commencement (POC)
- [ ] 2.4.1 Distinguish between POC (starting reference) and POB (property start)
- [ ] 2.4.2 Draw the tie line from the POC to the POB
- [ ] 2.4.3 Label both points correctly on the plot and in exports

### 2.5 Migrate to GitHub Actions for deployment ✅
- [x] 2.5.1 Replace Google Cloud Build with GitHub Actions workflow
- [x] 2.5.2 Better deployment visibility and history
- [x] 2.5.3 Automated CI/CD pipeline with proper testing
- [x] 2.5.4 Environment variable management through GitHub Secrets
- [x] 2.5.5 PR validation workflow
- [x] 2.5.6 Environment-specific deployments (prod/dev)

### 2.6 Calculate closure line
- [ ] 2.6.1 Calculate the distance and bearing from the end of the last line back to the beginning of line 1 (Point of Beginning)
- [ ] 2.6.2 Show closure error (how far off from perfect closure)
- [ ] 2.6.3 Display closure bearing and distance
- [ ] 2.6.4 Add visual indication on the plot
- [ ] 2.6.5 Include closure information in PDF export

### 2.7 Improve GPT prompt for different legal description types
- [ ] 2.7.1 Metes and bounds (current focus - bearings/distances)
- [ ] 2.7.2 Plat reference (Lot X, Block Y, Subdivision Z)
- [ ] 2.7.3 Government survey (Section, Township, Range)
- [ ] 2.7.4 Rectangular survey descriptions
- [ ] 2.7.5 Update property information section to show plat book/page references
- [ ] 2.7.6 Handle mixed descriptions that combine multiple types

### 2.8 Consider React Frontend Migration
- [ ] 2.8.1 Keep current Python backend (FastAPI/Flask) for AI processing
- [ ] 2.8.2 Build modern React frontend for better UX
- [ ] 2.8.3 Separate concerns: backend handles PDF processing, GPT analysis, DXF generation
- [ ] 2.8.4 Frontend handles interactive plotting, forms, file uploads
- [ ] 2.8.5 Better mobile responsiveness and modern UI components
- [ ] 2.8.6 TypeScript support for better development experience

## 3. 📊 Features

### 3.1 Add area calculation for closed polygons
- [ ] 3.1.1 Calculate polygon area

### 3.2 Support for curves and arcs in legal descriptions
- [ ] 3.2.1 Parse and render curves/arcs

### 3.3 Export to other CAD formats
- [ ] 3.3.1 Support AutoCAD and other formats

### 3.4 Add coordinate system transformations
- [ ] 3.4.1 Support different coordinate systems

## 4. 🎨 UI/UX

### 4.1 Improve debug architecture
- [ ] 4.1.1 Replace global DEBUG_MODE with page-specific debug controls
- [ ] 4.1.2 Add debug level granularity (off/basic/detailed/full)
- [ ] 4.1.3 Create context-aware debug display with cleaner UI using tabs
- [ ] 4.1.4 Implement user preference persistence for debug settings
- [ ] 4.1.5 Make main page default to clean UI, test pages to debug-enabled

### 4.2 Add Gmail login authentication
- [x] 4.2.1 User authentication: Secure access to the application using Google OAuth
- [ ] 4.2.2 Session management: Save user's current work across sessions
- [ ] 4.2.3 Personalization: Remember user preferences and settings
- [ ] 4.2.4 Usage tracking: Analytics on how the tool is being used
- [ ] 4.2.5 Project storage: Save and retrieve previous legal description analyses
- [ ] 4.2.6 User dashboard: Show history of processed documents
- [ ] 4.2.7 Sharing capabilities: Allow users to share projects with others

## 5. 🧪 Testing

### 5.1 Add unit tests for bearing calculations
- [ ] 5.1.1 Unit tests

### 5.2 Test with various PDF formats
- [ ] 5.2.1 PDF format testing

### 5.3 Validate closure calculations with known survey data
- [ ] 5.3.1 Closure validation

### 5.4 Load tests with large files and concurrent users
- [ ] 5.4.1 Performance testing

### 5.5 Security tests for input sanitization
- [ ] 5.5.1 Security testing

### 5.6 Memory leak tests with repeated operations
- [ ] 5.6.1 Memory testing

## 6. 🐛 Bug Fixes

### 6.1 Fix mobile axios errors after multiple uploads
- [ ] 6.1.1 Clear session state between documents to prevent memory buildup

### 6.2 Fix temporary file cleanup
- [ ] 6.2.1 PDF files saved with delete=False accumulate on disk (main.py:1078-1080)

### 6.3 Fix resource leak in DXF export
- [ ] 6.3.1 Files opened for reading but not properly closed in exception scenarios (main.py:677-678, 709-710, 1209-1210)

### 6.4 Fix session state race condition
- [ ] 6.4.1 Session ID generation not thread-safe for concurrent users (main.py:363-364)

### 6.5 Remove hardcoded file paths
- [ ] 6.5.1 Windows-specific path C:\dev\openai-key.json won't work cross-platform (main.py:17, utils/auth.py:25)

### 6.6 Fix API key exposure in logs
- [ ] 6.6.1 Debug prints reveal API key existence (main.py:24-28)

### 6.7 Standardize error handling
- [ ] 6.7.1 GPT parsing returns inconsistent error formats (main.py:523-525)

### 6.8 Fix file size calculation
- [ ] 6.8.1 getvalue() called multiple times inefficiently (main.py:1051, 1124)

### 6.9 Add database transaction handling
- [ ] 6.9.1 Operations not wrapped in transactions (utils/classification.py:35-45)

### 6.10 Optimize file processing
- [ ] 6.10.1 Files read multiple times unnecessarily (main.py:1079, 1787, 1733)

### 6.11 Pre-compile regex patterns
- [ ] 6.11.1 Patterns compiled on every function call (main.py:430-435)

## 7. 📝 Documentation

### 7.1 Create user guide
- [ ] 7.1.1 User guide

### 7.2 Add API documentation
- [ ] 7.2.1 API docs

### 7.3 Document surveying terminology used
- [ ] 7.3.1 Terminology docs

## 8. 🏗️ Technical Debt

### 8.1 Large monolithic main.py file
- [ ] 8.1.1 Refactor 1700+ line file

### 8.2 Mixed concerns
- [ ] 8.2.1 Separate UI, business logic, data processing

### 8.3 Limited error handling in some functions
- [ ] 8.3.1 Improve error handling

### 8.4 No automated testing
- [ ] 8.4.1 Add test suite

### 8.5 Hardcoded values scattered throughout
- [ ] 8.5.1 Move to config

### 8.6 Extract business logic into separate modules
- [ ] 8.6.1 Modularize code

### 8.7 Add type hints throughout
- [ ] 8.7.1 Type annotations

### 8.8 Implement proper logging
- [ ] 8.8.1 Structured logging

### 8.9 Add configuration management
- [ ] 8.9.1 Config system

### 8.10 Create data validation schemas
- [ ] 8.10.1 Validation schemas

---
*Last updated: 2025-12-04*
