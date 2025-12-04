# Legal Description Reader - TODO List

## Next Steps & Ideas

### 🎯 High Priority
- [ ] **Fix debug output clearing when drawing lines**:
  - Debug output from GPT processing gets cleared when clicking 'Draw Lines' because Streamlit reruns to update the plot
  - Need to modify plotting to update without full rerun - possibly using Plotly's update methods or in-place chart updates
  - Avoid session state changes that trigger rerun while preserving real-time debug context

- [ ] **Fix filename handling and login button formatting**:
  - Ensure filename field is always preserved in reasoning_data before database save
  - Put filename first in reasoning_data JSON object for better readability
  - Update login button to be compact without title for PDF preview section

- [ ] **Calculate closure line**: Calculate the distance and bearing from the end of the last line back to the beginning of line 1 (Point of Beginning)
  - Show closure error (how far off from perfect closure)
  - Display closure bearing and distance
  - Add visual indication on the plot
  - Include closure information in PDF export

- [ ] **Improve GPT prompt for different legal description types**:
  - Metes and bounds (current focus - bearings/distances)
  - Plat reference (Lot X, Block Y, Subdivision Z)
  - Government survey (Section, Township, Range)
  - Rectangular survey descriptions
  - Update property information section to show plat book/page references
  - Handle mixed descriptions that combine multiple types

### 🔄 Improvements
- [ ] **Fix classification database logging errors**:
  - Debug output currently has errors when saving to classification database
  - Need to investigate and fix database schema/connection issues
  - Once fixed, use classification logging for debug data that I can read
  - This will allow remote debugging without screenshots
- [ ] **Handle abstract bearings in parser**:
  - GPT sometimes returns "Northwesterly" instead of "N 45d 30m 15s W"
  - Parser skips these, causing inconsistent bearing counts
  - Convert abstract bearings (Northwesterly, Southeasterly, etc.) to approximate degrees
  - Example: Northwesterly = N 45° 0' 0" W
- [ ] Add validation for bearing inputs (ensure they make sense)
- [ ] Improve error handling for malformed PDFs
- [ ] Add support for more legal description formats
- [ ] Better monument/marker handling and display
- [ ] **Handle Point of Commencement (POC)**:
  - Distinguish between POC (starting reference) and POB (property start)
  - Draw the tie line from the POC to the POB
  - Label both points correctly on the plot and in exports
- [x] **Migrate to GitHub Actions for deployment**:
  - ✅ Replace Google Cloud Build with GitHub Actions workflow
  - ✅ Better deployment visibility and history
  - ✅ Automated CI/CD pipeline with proper testing
  - ✅ Environment variable management through GitHub Secrets
  - ✅ PR validation workflow
  - ✅ Environment-specific deployments (prod/dev)
- [ ] **Consider React Frontend Migration**:
  - Keep current Python backend (FastAPI/Flask) for AI processing
  - Build modern React frontend for better UX
  - Separate concerns: backend handles PDF processing, GPT analysis, DXF generation
  - Frontend handles interactive plotting, forms, file uploads
  - Better mobile responsiveness and modern UI components
  - TypeScript support for better development experience

### 📊 Features
- [ ] Add area calculation for closed polygons
- [ ] Support for curves and arcs in legal descriptions
- [ ] Export to other CAD formats (AutoCAD, etc.)
- [ ] Add coordinate system transformations

### 🎨 UI/UX
- [ ] **Improve debug architecture**:
  - Replace global DEBUG_MODE with page-specific debug controls
  - Add debug level granularity (off/basic/detailed/full)
  - Create context-aware debug display with cleaner UI using tabs
  - Implement user preference persistence for debug settings
  - Make main page default to clean UI, test pages to debug-enabled
- [ ] Add tooltips to explain surveying terms
- [ ] Improve mobile responsiveness
- [ ] Add keyboard shortcuts for common actions
- [ ] Better visual feedback during processing
- [ ] **Add Gmail login authentication**:
  - User authentication: Secure access to the application using Google OAuth
  - Session management: Save user's current work across sessions
  - Personalization: Remember user preferences and settings
  - Usage tracking: Analytics on how the tool is being used
  - Project storage: Save and retrieve previous legal description analyses
  - User dashboard: Show history of processed documents
  - Sharing capabilities: Allow users to share projects with others

### 🧪 Testing
- [ ] Add unit tests for bearing calculations
- [ ] Test with various PDF formats
- [ ] Validate closure calculations with known survey data

### 📝 Documentation
- [ ] Create user guide
- [ ] Add API documentation
- [ ] Document surveying terminology used

---
*Last updated: 2025-07-30*
