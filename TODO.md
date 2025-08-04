# Legal Description Reader - TODO List

## Next Steps & Ideas

### 🎯 High Priority
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
- [ ] Add validation for bearing inputs (ensure they make sense)
- [ ] Improve error handling for malformed PDFs
- [ ] Add support for more legal description formats
- [ ] Better monument/marker handling and display
- [ ] **Handle Point of Commencement (POC)**:
  - Distinguish between POC (starting reference) and POB (property start)
  - Draw the tie line from the POC to the POB
  - Label both points correctly on the plot and in exports

### 📊 Features
- [ ] Add area calculation for closed polygons
- [ ] Support for curves and arcs in legal descriptions
- [ ] Export to other CAD formats (AutoCAD, etc.)
- [ ] Add coordinate system transformations

### 🎨 UI/UX
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
