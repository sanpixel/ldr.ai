# LDR (Legal Description Reader) - Deployment & Architecture Guide

## Deployment Architecture

### Production Deployment: Google Cloud Platform
**Platform**: Google Cloud Run (Serverless Container Platform)  
**URL**: https://ldr.clocknumbers.com/  
**Region**: us-central1  
**Trigger**: Git push to main branch → Cloud Build → Cloud Run

### Deployment Flow
```
Git Push → Cloud Build (cloudbuild.yaml) → Docker Build → Container Registry → Cloud Run Deploy
```

1. **Developer pushes to Git repository**
2. **Cloud Build triggers automatically** (configured in GCP Console)
3. **Builds Docker image** using `Dockerfile`
4. **Pushes to Google Container Registry**: `gcr.io/$PROJECT_ID/ldr-ai:$COMMIT_SHA`
5. **Deploys to Cloud Run** service named `ldr-ai`

---

## Docker Configuration

### Dockerfile Details
- **Base Image**: `python:3.11-slim`
- **Working Directory**: `/app`
- **Port**: 8080 (Cloud Run standard)
- **System Dependencies**:
  - `tesseract-ocr` - OCR engine for PDF text extraction
  - `tesseract-ocr-eng` - English language pack
  - `poppler-utils` - PDF rendering utilities
- **Build Strategy**: Requirements copied first for Docker layer caching
- **Streamlit Config**: 
  - `STREAMLIT_SERVER_PORT=8080`
  - `STREAMLIT_SERVER_ADDRESS=0.0.0.0`

### .dockerignore
Excludes from Docker build:
- Git files (`.git`, `.gitignore`)
- Environment files (`.env`, `*.env`)
- Python cache (`__pycache__`, `*.pyc`)
- IDE files (`.vscode`, `.idea`, `.cursor`)
- Generated DXF files
- Documentation (README, TODO)

---

## Cloud Build Configuration (cloudbuild.yaml)

### Build Steps
1. **Build**: `docker build -t gcr.io/$PROJECT_ID/ldr-ai:$COMMIT_SHA`
2. **Push**: Push image to Google Container Registry
3. **Deploy**: Deploy to Cloud Run with configuration

### Cloud Run Deployment Settings
- **Service Name**: `ldr-ai`
- **Region**: `us-central1`
- **Platform**: `managed` (fully managed Cloud Run)
- **Access**: `--allow-unauthenticated` (public access)
- **Resources**:
  - Memory: `2Gi`
  - CPU: `2`
  - Max Instances: `10` (autoscaling)
- **Environment Variables** (set via Cloud Build substitutions):
  - `OPENAI_API_KEY=${_OPENAI_API_KEY}`
  - `GOOGLE_DRIVE_API_KEY=${_GOOGLE_DRIVE_API_KEY}`
- **Tag**: `${_ENV_NAME}` (for environment tracking)

### Missing from cloudbuild.yaml (Set in Cloud Run Console)
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `ENVIRONMENT=production`
- `APP_URL=https://ldr.clocknumbers.com/`

---

## Environment Variables

### Required (Core Functionality)
| Variable | Purpose | Source |
|----------|---------|--------|
| `OPENAI_API_KEY` | GPT-4 text processing & classification | Cloud Build substitution |
| `SUPABASE_URL` | Database connection | Cloud Run console (hardcoded: `https://xvlzjyjqqgfpcxqnplds.supabase.co`) |
| `SUPABASE_ANON_KEY` | Database authentication | Cloud Run console |

### Optional (Enhanced Features)
| Variable | Purpose | Source |
|----------|---------|--------|
| `GOOGLE_DRIVE_API_KEY` | Google Drive PDF integration | Cloud Build substitution |
| `SUPABASE_SERVICE_ROLE_KEY` | Admin database operations | Cloud Run console |
| `ENVIRONMENT` | Environment detection (production/development) | Cloud Run console |
| `APP_URL` | OAuth redirect URL | Cloud Run console |

### Unused (Legacy)
- `GOOGLE_CLIENT_ID` - Not used (OAuth via Supabase)
- `GOOGLE_CLIENT_SECRET` - Not used (OAuth via Supabase)

---

## Local Development Configuration

### Streamlit Config (.streamlit/config.toml)
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
enableCORS = true
enableXsrfProtection = false

[browser]
serverAddress = "0.0.0.0"
serverPort = 5000
gatherUsageStats = false
```

### Local Port vs Production Port
- **Local Development**: Port 5000 (Streamlit default override)
- **Production (Cloud Run)**: Port 8080 (Cloud Run standard)

### Environment Variable Loading (utils/auth.py)
```python
# Priority order:
1. Environment variables (os.getenv)
2. Fallback: Local JSON file at C:\dev\openai-key.json (legacy, not recommended)
```

---

## Python Dependencies (requirements.txt)

### Core Framework
- `streamlit` - Web application framework

### Data Processing
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `plotly` - Interactive visualizations

### PDF & OCR
- `pytesseract` - Python wrapper for Tesseract OCR
- `pdf2image` - PDF to image conversion
- `reportlab` - PDF generation

### AI & APIs
- `openai` - GPT-4 API client
- `supabase>=2.0.0` - Database client

### Authentication
- `python-jose[cryptography]` - JWT handling
- `streamlit-oauth` - OAuth flows
- `streamlit-js` - JavaScript execution in Streamlit
- `streamlit-url-fragment` - URL fragment handling

### CAD & Drawing
- `ezdxf` - DXF file generation for AutoCAD

### Utilities
- `python-dotenv` - .env file loading
- `requests` - HTTP client

---

## Database: Supabase PostgreSQL

### Connection Details
- **URL**: `https://xvlzjyjqqgfpcxqnplds.supabase.co`
- **Client**: Initialized globally in `utils/auth.py`
- **Authentication**: Anon key for public operations

### Classification Data Table
**Table**: `classification_data`

**Schema** (from `classification_table.sql`):
```sql
CREATE TABLE classification_data (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  reasoning_data JSONB NOT NULL,
  
  -- Generated columns for fast filtering
  classification TEXT GENERATED ALWAYS AS (reasoning_data->>'classification') STORED,
  confidence TEXT GENERATED ALWAYS AS (reasoning_data->>'confidence') STORED,
  created_at TIMESTAMPTZ GENERATED ALWAYS AS ((reasoning_data->>'timestamp')::timestamptz) STORED,
  
  inserted_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Indexes**:
- `idx_classification_data_classification` - Fast classification filtering
- `idx_classification_data_confidence` - Fast confidence filtering
- `idx_classification_data_created_at` - Time-based queries
- `idx_classification_data_inserted_at` - Insertion order
- `idx_classification_data_reasoning` - GIN index for JSON queries

**Constraints**:
- Classification must be: `explicit_bearings`, `abstract_bearings`, `external_ref`
- Confidence must be: `high`, `medium`, `low`

**Row Level Security (RLS)**:
- Enabled with policies for authenticated users
- Anonymous read access allowed

### Database Operations (utils/classification.py)
- `save_classification_data()` - Insert reasoning data
- `load_all_classification_data()` - Fetch all records
- `get_filtered_classification_data()` - Filtered queries
- `clear_all_classification_data()` - Delete all (admin only)
- `test_database_connection()` - Connection validation

---

## Authentication: Supabase OAuth

### OAuth Provider
- **Provider**: Google (via Supabase Auth)
- **Implementation**: `utils/auth.py`
- **Session Storage**: Streamlit session state + browser local storage

### OAuth Flow
1. User clicks "Sign in with Google"
2. Redirects to Google OAuth
3. Returns with authorization code
4. Code exchanged for Supabase session
5. User metadata stored in `st.session_state.user`

### Redirect URLs (Configured in Supabase)
- **Local**: `http://localhost:5000/`
- **Production**: `https://ldr.clocknumbers.com/`

### Environment Detection
```python
if os.getenv("ENVIRONMENT") == "production":
    redirect_to = os.getenv("APP_URL", "https://ldr.clocknumbers.com")
else:
    redirect_to = "http://localhost:5000"
```

---

## Application Structure

### Main Application
- **Entry Point**: `main.py`
- **Framework**: Streamlit
- **Pages**: Multi-page app in `pages/` directory

### Key Modules
- `utils/auth.py` - Supabase authentication & client
- `utils/classification.py` - Database operations for classification data
- `utils/st_local_storage.py` - Browser local storage interface

### Pages
- `pages/reasoning.py` - Classification reasoning dashboard

---

## Deployment History

### Last Known Good Deploy
- **Deploy #76**: Commit `d3bf195`
- **Date**: 2025-08-31
- **Status**: Stable before OAuth changes

### Recent Changes
- OAuth authentication implementation
- Fixed redirect URL (port 5000 vs 8501)
- Simplified login UI (Google only)
- Environment-aware URL handling

---

## Rollback Procedure

If deployment breaks:
```bash
# Option 1: Revert recent commits
git revert HEAD~n  # where n = number of commits to revert

# Option 2: Hard reset to last good deploy
git reset --hard d3bf195
git push --force-with-lease origin main
```

Cloud Build will automatically trigger and redeploy.

---

## Security Notes

### CORS Configuration
- **Enabled**: `enableCORS = true` in `.streamlit/config.toml`
- **XSRF Protection**: Disabled (`enableXsrfProtection = false`)
- **Reason**: Required for PDF file uploads across browsers
- **Risk**: Consider restricting CORS in production

### Secrets Management
- **Production**: Environment variables in Cloud Run console
- **Local**: `.env` file (gitignored) or local JSON file
- **Never commit**: API keys, database credentials

### Public Access
- Cloud Run service allows unauthenticated access
- Application-level authentication via Supabase OAuth
- Database protected by Row Level Security policies

---

## Monitoring & Debugging

### Cloud Run Logs
Access via GCP Console:
- Cloud Run → ldr-ai service → Logs
- Filter by severity, timestamp, request ID

### Streamlit Debug Mode
Enable in app with session state:
```python
st.session_state['debug_enabled'] = True
```

### Database Connection Test
```python
from utils.classification import test_database_connection
if not test_database_connection():
    st.error("Database connection failed")
```

---

## Common Issues

### Supabase Auto-Pause
- **Issue**: Free tier pauses after inactivity
- **Symptom**: Database connection failures
- **Solution**: Unpause in Supabase dashboard

### Port Mismatches
- **Local**: Port 5000 (Streamlit config)
- **Production**: Port 8080 (Dockerfile)
- **OAuth**: Must match redirect URL configuration

### Missing Environment Variables
- Check Cloud Run console → ldr-ai → Variables & Secrets
- Verify Cloud Build substitution variables
- Ensure Supabase keys are set

---

## Development Workflow

### Local Development
1. Clone repository
2. Copy `.env.example` to `.env`
3. Fill in API keys
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `streamlit run main.py`
6. Access: `http://localhost:5000`

### Deployment
1. Make changes locally
2. Test thoroughly
3. Commit and push to Git
4. Cloud Build automatically deploys
5. Verify at `https://ldr.clocknumbers.com/`

### Testing OAuth
- Local: Use `http://localhost:5000/` redirect
- Production: Use `https://ldr.clocknumbers.com/` redirect
- Ensure both URLs configured in Supabase

---

## Performance Optimization

### Docker Layer Caching
- Requirements copied before application code
- Reduces rebuild time when only code changes

### Database Indexing
- Generated columns for fast filtering
- GIN index for complex JSON queries
- Proper index on timestamp fields

### Cloud Run Autoscaling
- Min instances: 0 (scales to zero)
- Max instances: 10
- 2Gi memory, 2 CPU per instance

---

## Future Considerations

### Potential Improvements
- Add health check endpoint for monitoring
- Implement structured logging
- Add performance metrics collection
- Consider CDN for static assets
- Implement rate limiting for API calls

### Security Enhancements
- Restrict CORS to specific origins
- Enable XSRF protection with proper token handling
- Implement API key rotation strategy
- Add request logging and audit trail

---

## Quick Reference

### URLs
- **Production**: https://ldr.clocknumbers.com/
- **Supabase**: https://xvlzjyjqqgfpcxqnplds.supabase.co
- **Container Registry**: gcr.io/$PROJECT_ID/ldr-ai

### Ports
- **Local**: 5000
- **Production**: 8080

### Key Files
- `Dockerfile` - Container definition
- `cloudbuild.yaml` - Build & deploy config
- `.streamlit/config.toml` - Streamlit settings
- `requirements.txt` - Python dependencies
- `utils/auth.py` - Authentication & DB client
- `utils/classification.py` - Database operations

### Commands
```bash
# Local development
streamlit run main.py

# Docker build (local testing)
docker build -t ldr-ai .
docker run -p 8080:8080 ldr-ai

# Git deployment
git add .
git commit -m "Description"
git push origin main
```
