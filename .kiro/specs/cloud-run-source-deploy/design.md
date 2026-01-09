# Design Document: Cloud Run Deploy from Source Without Build

## Overview

This design describes the migration from Cloud Build container image building to Cloud Run's "deploy from source without build" feature. The new approach packages the application source and dependencies into a .tar.gz archive, uploads it to Cloud Storage, and deploys it directly to Cloud Run using a base image. This eliminates the container build step, resulting in faster deployments while maintaining identical functionality.

## Architecture

### Current Deployment Flow
```
Git Push → GitHub Actions → Cloud Build (builds container) → Cloud Run
```

### New Deployment Flow
```
Git Push → GitHub Actions → Create Archive → Upload to GCS → Cloud Run Deploy (--no-build)
```

### Key Changes
1. **No Dockerfile execution**: The Dockerfile is no longer used during deployment
2. **Pre-packaged dependencies**: All Python dependencies are installed locally before archiving
3. **Base image runtime**: Cloud Run uses a pre-built Python 3.11 base image
4. **Direct source deployment**: Source code runs directly on the base image

## Components and Interfaces

### 1. Archive Creation Component

**Purpose**: Package application source and dependencies into a deployable archive

**Inputs**:
- Application source directory (root of repository)
- requirements.txt (Python dependencies)

**Process**:
1. Install all Python dependencies locally to a `vendor` directory
2. Create a .tar.gz archive containing:
   - All Python source files (main.py, pages/, utils/)
   - All configuration files (.streamlit/config.toml)
   - Installed dependencies (vendor directory)
   - Exclude: .git, __pycache__, .env, .vscode, .cursor, etc.
3. Verify archive size is under 250 MiB
4. Output: `ldr-ai-source-{timestamp}.tar.gz`

**Constraints**:
- Archive must be self-contained (all dependencies included)
- Archive size must not exceed 250 MiB
- Must exclude sensitive files (.env, API keys)
- Must exclude unnecessary files (git, cache, IDE files)

### 2. Cloud Storage Upload Component

**Purpose**: Upload the source archive to Cloud Storage for Cloud Run access

**Inputs**:
- Source archive file path
- Cloud Storage bucket name
- GCP project credentials

**Process**:
1. Authenticate with GCP using service account key
2. Upload archive to Cloud Storage bucket
3. Verify upload completion
4. Return Cloud Storage path for deployment

**Constraints**:
- Bucket must exist and be accessible
- Service account must have storage.objects.create permission
- Upload must complete successfully before proceeding to deployment

### 3. Cloud Run Deployment Component

**Purpose**: Deploy the source archive to Cloud Run using --no-build flag

**Inputs**:
- Cloud Storage path to source archive
- Cloud Run service name (ldr-ai)
- Base image (python311)
- Startup command (streamlit run main.py)
- Environment variables

**Process**:
1. Authenticate with GCP
2. Execute gcloud command with --no-build flag:
   ```
   gcloud beta run deploy ldr-ai \
     --source gs://bucket/archive.tar.gz \
     --region us-central1 \
     --no-build \
     --base-image python311 \
     --command streamlit \
     --args main.py \
     --set-env-vars PORT=8080,ENVIRONMENT=production,...
   ```
3. Wait for deployment to complete
4. Verify service is running and accessible
5. Return deployment status

**Constraints**:
- Must use python311 base image (supports Python 3.11)
- Must set PORT=8080 for Cloud Run
- Must preserve all existing environment variables
- Must maintain same OAuth redirect URLs

### 4. GitHub Actions Workflow Component

**Purpose**: Orchestrate the entire deployment process on git push

**Inputs**:
- Git branch (main, dev, prod)
- GitHub Actions secrets (GCP_SA_KEY_RAW, environment variables)

**Process**:
1. Checkout code
2. Authenticate with GCP using service account key
3. Install Python dependencies
4. Create source archive
5. Upload to Cloud Storage
6. Deploy to Cloud Run
7. Report status

**Constraints**:
- Must trigger on push to main, dev, prod branches
- Must use GitHub Actions secrets for credentials
- Must fail fast if any step fails
- Must provide clear error messages

## Data Models

### Archive Metadata
```
{
  "filename": "ldr-ai-source-20240109-143022.tar.gz",
  "created_at": "2024-01-09T14:30:22Z",
  "size_bytes": 45000000,
  "gcs_path": "gs://ldr-deploy-bucket/ldr-ai-source-20240109-143022.tar.gz",
  "git_commit": "abc123def456",
  "git_branch": "main"
}
```

### Deployment Status
```
{
  "deployment_id": "deploy-20240109-143022",
  "status": "success|failed|in_progress",
  "service": "ldr-ai",
  "region": "us-central1",
  "timestamp": "2024-01-09T14:30:22Z",
  "duration_seconds": 120,
  "error_message": null
}
```

## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Archive Contains All Source Files

**For any** application source directory, the created archive should contain all Python source files (main.py, pages/, utils/), configuration files (.streamlit/config.toml), and all installed dependencies from requirements.txt.

**Validates: Requirements 1.1, 1.2, 1.5**

### Property 2: Archive Excludes Unnecessary Files

**For any** application source directory, the created archive should NOT contain unnecessary files such as .git, __pycache__, .env, .vscode, .cursor, or other cache/IDE files.

**Validates: Requirements 1.3**

### Property 3: Archive Size Under Limit

**For any** valid application source, the created archive should not exceed 250 MiB in size.

**Validates: Requirements 1.4**

### Property 4: Archive Upload Succeeds

**For any** valid source archive, uploading it to Cloud Storage should result in the file being accessible at the specified GCS path.

**Validates: Requirements 2.1, 2.2**

### Property 5: Deployment Command Correctness

**For any** deployment to Cloud Run, the gcloud command should include the --no-build flag, specify python311 base image, set PORT=8080, and use the correct startup command (streamlit run main.py).

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

### Property 6: Deployed Service Accessibility

**For any** source archive deployed to Cloud Run, the deployed service should be running and respond to HTTP requests.

**Validates: Requirements 3.5**

### Property 7: Environment Variables Preserved

**For any** set of environment variables configured before deployment, all variables should be accessible to the running application after deployment via source without build.

**Validates: Requirements 5.1, 5.2, 5.3**

### Property 8: Functional Equivalence After Migration

**For any** request to the application, the response should be identical whether the application is deployed via Cloud Build or via source without build (same Streamlit app, same OAuth, same database connections, same API integrations).

**Validates: Requirements 6.1, 6.2, 6.3, 6.4**

### Property 9: Deployment Idempotence

**For any** source archive deployed to Cloud Run, deploying the same archive multiple times should result in the same running service state (same version, same configuration, same environment variables).

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

### Property 10: Workflow Automation

**For any** git push to main/dev/prod branches, the GitHub Actions workflow should automatically trigger and execute all deployment steps (archive creation, upload, deployment).

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**

### Property 11: Deployment Performance Improvement

**For any** deployment via source without build compared to Cloud Build, the deployment should complete faster (measured in seconds).

**Validates: Requirements 7.1, 7.2**

### Property 12: Rollback Capability

**For any** failed deployment, the previous working version should remain running and accessible, and the system should allow rollback to a previous deployment.

**Validates: Requirements 8.1, 8.2, 8.3**

## Error Handling

### Archive Creation Errors
- **Missing requirements.txt**: Fail with clear error message
- **Dependency installation failure**: Report which package failed and why
- **Archive size exceeds 250 MiB**: Fail with size information and suggestions
- **File permission errors**: Fail with file path and permission details

### Cloud Storage Upload Errors
- **Authentication failure**: Fail with GCP credential error
- **Bucket not found**: Fail with bucket name and project ID
- **Upload timeout**: Retry up to 3 times before failing
- **Insufficient permissions**: Fail with required IAM role

### Cloud Run Deployment Errors
- **Invalid base image**: Fail with available base images
- **Service not found**: Create new service or fail with details
- **Deployment timeout**: Report timeout and suggest checking Cloud Run logs
- **Environment variable errors**: Report which variable failed and why

### GitHub Actions Errors
- **Checkout failure**: Fail with git error details
- **Authentication failure**: Fail with GCP credential error
- **Any step failure**: Stop workflow and report error in GitHub Actions UI

## Testing Strategy

### Unit Tests
- Test archive creation with various source structures
- Test archive size calculation
- Test file exclusion patterns (.gitignore, cache, etc.)
- Test environment variable handling
- Test deployment command generation

### Property-Based Tests
- **Property 1**: Generate random source directories and verify archive contains all necessary files
- **Property 2**: Generate various source sizes and verify archive stays under 250 MiB
- **Property 3**: Deploy same archive multiple times and verify identical service state
- **Property 4**: Generate random environment variable sets and verify preservation
- **Property 5**: Compare responses from Cloud Build vs source deploy (integration test)
- **Property 6**: Simulate git pushes and verify workflow triggers

### Integration Tests
- End-to-end deployment from git push to running service
- Verify OAuth still works after deployment
- Verify database connections work after deployment
- Verify API integrations work after deployment
- Test rollback to previous version

### Manual Testing
- Deploy to dev branch and verify functionality
- Deploy to prod branch and verify no downtime
- Test OAuth flow end-to-end
- Verify all environment variables are accessible
- Check Cloud Run logs for any errors

## Deployment Checklist

Before deploying to production:
1. ✓ Archive creation tested with current source
2. ✓ Archive size verified under 250 MiB
3. ✓ Cloud Storage bucket created and accessible
4. ✓ GitHub Actions workflow tested on dev branch
5. ✓ Environment variables verified in Cloud Run console
6. ✓ OAuth redirect URLs verified in Supabase
7. ✓ Rollback procedure documented
8. ✓ Monitoring and logging verified
