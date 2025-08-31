# Deployment Notes

## Last Known Good Deploy

**Deploy #76**: Commit `d3bf195` pushed by `sanpixel`
- Status: ✅ **LAST GOOD DEPLOY**
- Date: 2025-08-31
- Notes: This is the last stable deployment before OAuth authentication changes

## Recent Updates

### OAuth Authentication Implementation ✅ COMPLETED
- **Issue**: OAuth redirect URL was pointing to wrong port (8501 instead of 5000)
- **Fix**: Updated `utils/auth.py` with proper OAuth flow implementation
- **Status**: ✅ Ready for deployment testing
- **Configuration**: Added both `http://localhost:5000/` and `https://ldr.clocknumbers.com/` to Supabase redirect URLs

### Authentication Changes Implemented
1. **Simplified Login UI**: 
   - Removed GitHub login option
   - Shows only Google sign-in button
   - Button appears in main content when not authenticated
   - User menu appears in sidebar when authenticated

2. **Fixed OAuth Flow**:
   - Corrected redirect URL from port 8501 to 5000 for local development
   - Fixed `exchange_code_for_session` method to use proper `CodeExchangeParams`
   - Added comprehensive error handling and debugging
   - Proper session management and user data storage

3. **Environment Configuration**:
   - Local development: `http://localhost:5000/`
   - Production: Falls back to `https://ldr.clocknumbers.com/`
   - Respects `ENVIRONMENT=production` and `APP_URL` environment variables

4. **Updated main.py**:
   - OAuth callback handling at app startup
   - Conditional UI rendering based on authentication state
   - Welcome messages and proper session management

## Rollback Instructions

If current changes break the deployment:
```bash
git revert HEAD~n  # where n is number of commits since d3bf195
# OR
git reset --hard d3bf195
git push --force-with-lease origin main
```

## Next Steps
1. Test OAuth authentication locally with port 5000
2. Update Supabase project settings to allow localhost:5000 redirect
3. Verify authentication flow works end-to-end
4. Deploy to Cloud Run once local testing is complete
