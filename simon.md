# Development Notes & Questions for Simon

## Current Session Notes

### Issues Resolved
1. ✅ **Fallback GPT Prompt**: Embedded current working prompt as fallback in `main.py` for Cloud Run reliability
2. ✅ **Debug UI Consolidation**: All debug output now uses expandable `st.expander()` components
3. ✅ **Bullet Point Parsing Fix**: Fixed alternatives section parsing in GPT reasoning output

### React Migration Planning
- ✅ **Comprehensive Migration Guide**: Created detailed 4-week migration plan in `REACT_MIGRATION_GUIDE.md`
- ✅ **Architecture Design**: Proposed React frontend + FastAPI backend + Supabase architecture
- ✅ **Technology Stack**: Recommended modern stack (React 18, TypeScript, Vite, Tailwind, Zustand)
- ✅ **Database Schema**: Designed Supabase tables for data persistence
- ✅ **Deployment Strategy**: Cloud Run for backend, Vercel/Netlify for frontend

## Questions for Simon

### React Migration Priority
1. Should we proceed with React migration or focus on other improvements first?
2. Any preference between Vercel vs Netlify for frontend hosting?
3. How important is mobile optimization for your use case?

### Debug Features
1. Are the current debug expandable sections working well for you?
2. What specific debug information is most valuable during troubleshooting?
3. Should we implement real-time debug logging in the React version?

### Data Persistence
1. When we implement Supabase, do you want to save all user sessions?
2. Should we implement user accounts or keep it anonymous for now?
3. Any requirements for data retention/deletion policies?

## Technical Decisions Pending

### Prompt File Issue
- Current Issue: `bearings_prompt.txt` file reading is inconsistent on Cloud Run
- Solution Implemented: Embedded fallback prompt in code
- Question: Should we move entirely to embedded prompts or keep file-based approach?

### Debug Mode Control
- Current: Single `DEBUG_MODE` flag controls all debug output
- Consideration: Should we have granular debug controls (OCR, GPT, Parsing, etc.)?
- Implementation: Could use environment variables like `DEBUG_OCR=true DEBUG_GPT=false`

### Error Handling Strategy
- Current: Basic try/catch with Streamlit error display
- React Migration: Need comprehensive error boundaries and user-friendly messages
- Question: What level of error detail should users see vs. what should be logged?

## Development Workflow Notes

### Version Control Strategy
- Current: Single repository with version files in `/pages`
- Consideration: For React migration, should we use separate repos or monorepo?
- Benefits of monorepo: Shared types, easier deployment coordination
- Benefits of separate: Independent versioning, clearer separation

### Testing Approach
- Current: Manual testing with debug buttons
- Proposed: Automated testing with Jest/pytest
- Question: What level of test coverage is needed? (unit tests, integration tests, e2e?)

### Deployment Pipeline
- Current: Manual deployment
- Proposed: GitHub Actions for both frontend and backend
- Question: Do you want staging environments or direct to production?

## Feature Requests & Ideas

### User Experience Improvements
- [ ] Progress indicators for PDF processing
- [ ] Drag-and-drop file upload
- [ ] Keyboard shortcuts for common actions
- [ ] Undo/redo for bearing edits
- [ ] Bulk bearing import/export
- [ ] Template saving for common property types

### Technical Improvements
- [ ] Offline mode with service workers
- [ ] Real-time collaboration (multiple users editing)
- [ ] API rate limiting and quotas
- [ ] Automated backup/restore
- [ ] Performance monitoring and alerts

### Integration Possibilities
- [ ] Google Drive/Dropbox integration
- [ ] Email PDF reports
- [ ] CAD software plugins
- [ ] Mobile app (React Native)
- [ ] WhatsApp/SMS notifications

## Lessons Learned

### Cloud Run Considerations
- File system is ephemeral - don't rely on local files
- Environment variables are more reliable than file configs
- Container restarts can happen frequently
- Logging to external services is essential

### Streamlit Limitations Discovered
- Complex state management becomes unwieldy
- Limited mobile responsiveness
- Debug information display is verbose
- No real-time updates without page refresh
- Difficult to customize UI/UX

### GPT Integration Insights
- Response format consistency is crucial
- Fallback prompts are essential for reliability
- Debug information helps troubleshoot parsing issues
- Classification reasoning is valuable for improving prompts

## Next Actions Required

### Immediate (This Week)
1. Get Simon's feedback on React migration timeline
2. Test current fixes on Cloud Run deployment
3. Verify PDF preview functionality is restored
4. Confirm debug UI improvements work as expected

### Short Term (Next 2 Weeks)
1. If approved, begin FastAPI backend development
2. Set up Supabase database schema
3. Create API endpoint specifications
4. Begin React frontend project structure

### Medium Term (Month 1-2)
1. Complete React migration
2. Implement user authentication
3. Add comprehensive testing
4. Set up monitoring and alerting

## Code Quality Notes

### Current Technical Debt
- [ ] Large monolithic `main.py` file (1700+ lines)
- [ ] Mixed concerns (UI, business logic, data processing)
- [ ] Limited error handling in some functions
- [ ] No automated testing
- [ ] Hardcoded values scattered throughout

### Improvement Opportunities
- [ ] Extract business logic into separate modules
- [ ] Add type hints throughout
- [ ] Implement proper logging
- [ ] Add configuration management
- [ ] Create data validation schemas

---

*This file is updated during development sessions to track decisions, questions, and progress.*
