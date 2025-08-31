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

## OAuth Implementation Details

### Authentication Flow
1. **Unauthenticated State**: Shows centered Google sign-in button in main content
2. **OAuth Process**: 
   - User clicks "🟢 Sign in with Google" 
   - Redirects to Google OAuth with proper redirect URL
   - Returns with authorization code
   - Code exchanged for session using `CodeExchangeParams`
3. **Authenticated State**: Google button disappears, user menu appears in sidebar

### Technical Implementation
- **Library**: Supabase Auth with Python client v2.18.1
- **Method**: `exchange_code_for_session(CodeExchangeParams(auth_code=code))`
- **Session Storage**: Streamlit session state with user metadata
- **Error Handling**: Comprehensive debugging and fallback mechanisms

### Environment Configuration
```python
# Local Development
redirect_to = "http://localhost:5000/"

# Production (when ENVIRONMENT=production)
redirect_to = os.getenv("APP_URL", "https://ldr.clocknumbers.com/")
```

### Required Supabase Configuration
- ✅ Redirect URLs configured: `http://localhost:5000/`, `https://ldr.clocknumbers.com/`
- ✅ Google OAuth provider enabled
- ✅ Environment variables: `SUPABASE_URL`, `SUPABASE_ANON_KEY`

## Deployment Readiness
- ✅ Code committed and pushed (commit: 39b1bb6)
- ✅ OAuth redirect URLs configured in Supabase
- ✅ Environment-aware URL handling
- ✅ Comprehensive error handling and debugging
- ✅ Rollback plan documented (Deploy #76, commit d3bf195)

## Testing Status
- ⚠️ Local OAuth still showing "Invalid API key" error during code exchange
- ✅ OAuth callback receiving authorization code correctly 
- ✅ Redirect URL fixed (port 5000)
- ⚠️ May need Supabase OAuth provider configuration check

**Ready for Cloud Run deployment to test production environment.**
