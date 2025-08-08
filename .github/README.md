# GitHub Actions CI/CD

This directory contains GitHub Actions workflows for the LDR.ai Legal Description Reader application.

## Workflows

### 🚀 `deploy.yml` - Deployment Workflow
Automatically deploys the Streamlit application to Google Cloud Run.

**Triggers:**
- Push to `main` → Production deployment (`ldr-ai`)
- Push to `prod` → Production deployment (`ldr-ai`) 
- Push to `dev` → Development deployment (`ldr-ai-dev`)
- Manual trigger via workflow_dispatch

**Requirements:**
- `GCP_PROJECT_ID` secret
- `GCP_SA_KEY` secret (base64 encoded service account key)
- `OPENAI_API_KEY` secret

### 🧪 `pr-validation.yml` - Pull Request Validation
Validates pull requests before merging.

**Checks:**
- Python syntax validation
- Docker build test
- Dependency installation
- Required files presence
- Import testing

**Triggers:**
- Pull requests to `main`, `dev`, or `prod` branches

## Required GitHub Secrets

Set these in your repository: **Settings > Secrets and variables > Actions > Repository secrets**

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `GCP_PROJECT_ID` | Google Cloud project ID | `warp-ratio` |
| `GCP_SA_KEY` | Base64 encoded service account JSON key | `ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsLi4u` |
| `OPENAI_API_KEY` | OpenAI API key for GPT processing | `sk-...` |

## Deployment Environments

| Branch | Environment | Service Name | URL |
|--------|-------------|--------------|-----|
| `main` | Production | `ldr-ai` | https://ldr.ai |
| `prod` | Production | `ldr-ai` | https://ldr.ai |
| `dev` | Development | `ldr-ai-dev` | Auto-generated |

## Service Account Setup

The GitHub Actions workflow uses a service account with these roles:
- `roles/run.admin` - Deploy to Cloud Run
- `roles/storage.admin` - Push to Container Registry
- `roles/iam.serviceAccountUser` - Use service accounts

## Migration from Cloud Build

This GitHub Actions setup **replaces** the previous Google Cloud Build deployment (`cloudbuild.yaml`). 

**Benefits:**
- ✅ Better visibility into deployment process
- ✅ Integrated with GitHub PR workflow  
- ✅ Environment-specific deployments
- ✅ Manual deployment triggers
- ✅ Detailed deployment logs

## Manual Deployment

You can trigger deployments manually:
1. Go to **Actions** tab in GitHub
2. Select **Deploy to Google Cloud Run** workflow
3. Click **Run workflow** 
4. Select branch and click **Run workflow**

## Troubleshooting

### Common Issues:

**"Service account key invalid"**
- Verify `GCP_SA_KEY` is properly base64 encoded
- Ensure service account has required roles

**"Docker push failed"** 
- Check Google Container Registry permissions
- Verify project ID is correct

**"Cloud Run deployment failed"**
- Check service account has `roles/run.admin`
- Verify region and resource limits

### Debug Tips:

1. Check workflow logs in **Actions** tab
2. Verify all secrets are set correctly
3. Test service account permissions locally:
   ```bash
   echo "$GCP_SA_KEY" | base64 -d > key.json
   gcloud auth activate-service-account --key-file key.json
   gcloud run services list
   ```
