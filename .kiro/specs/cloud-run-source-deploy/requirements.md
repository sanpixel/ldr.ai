# Requirements Document: Cloud Run Deploy from Source Without Build

## Introduction

Migrate the LDR application from traditional Cloud Build container image building to Cloud Run's "deploy from source without build" feature. This approach bypasses the container build step by uploading a pre-packaged source archive directly to Cloud Storage, allowing Cloud Run to run it on a base image. This results in dramatically faster deployment times while maintaining the same functionality.

## Glossary

- **Cloud Run**: Google Cloud's serverless container platform
- **Deploy from Source**: Cloud Run feature that runs source code directly on a base image without building a container
- **Source Archive**: A .tar.gz file containing the application source and dependencies
- **Base Image**: Pre-built runtime image (e.g., python311) provided by Google Cloud
- **GitHub Actions**: CI/CD workflow that triggers on git push
- **Cloud Storage**: Google Cloud object storage where source archives are uploaded
- **Deployment**: The process of getting code from git to running on Cloud Run

## Requirements

### Requirement 1: Archive Preparation

**User Story:** As a developer, I want the application source to be packaged into a deployable archive, so that it can be uploaded to Cloud Storage without requiring a build step.

#### Acceptance Criteria

1. WHEN the deployment process begins, THE System SHALL create a .tar.gz archive containing all application source files
2. WHEN the archive is created, THE System SHALL include all Python dependencies from requirements.txt
3. WHEN the archive is created, THE System SHALL exclude unnecessary files (git, cache, environment files, IDE files)
4. WHEN the archive is created, THE System SHALL ensure the total size is under 250 MiB
5. WHEN the archive is ready, THE System SHALL verify it contains all necessary files for the application to run

### Requirement 2: Cloud Storage Upload

**User Story:** As a deployment system, I want to upload the source archive to Cloud Storage, so that Cloud Run can access it for deployment.

#### Acceptance Criteria

1. WHEN an archive is ready for deployment, THE System SHALL upload it to a designated Cloud Storage bucket
2. WHEN the upload completes, THE System SHALL verify the archive exists in Cloud Storage
3. WHEN the upload fails, THE System SHALL retry the operation or report the error

### Requirement 3: Cloud Run Deployment

**User Story:** As a developer, I want the application to deploy from the source archive to Cloud Run, so that the service is updated with the latest code.

#### Acceptance Criteria

1. WHEN a source archive is available in Cloud Storage, THE System SHALL deploy it to Cloud Run using the --no-build flag
2. WHEN deploying, THE System SHALL use the python311 base image
3. WHEN deploying, THE System SHALL specify the correct startup command (streamlit run main.py)
4. WHEN deploying, THE System SHALL set the PORT environment variable to 8080
5. WHEN deployment completes, THE System SHALL verify the service is running and accessible

### Requirement 4: GitHub Actions Integration

**User Story:** As a developer, I want the deployment process to be automated via GitHub Actions, so that pushing code triggers the deployment without manual intervention.

#### Acceptance Criteria

1. WHEN code is pushed to main/dev/prod branches, THE GitHub Actions workflow SHALL trigger automatically
2. WHEN the workflow runs, THE System SHALL create the source archive
3. WHEN the workflow runs, THE System SHALL upload the archive to Cloud Storage
4. WHEN the workflow runs, THE System SHALL deploy to Cloud Run
5. WHEN deployment succeeds, THE System SHALL report success in the GitHub Actions log
6. WHEN deployment fails, THE System SHALL report the error and stop the workflow

### Requirement 5: Environment Variables and Secrets

**User Story:** As a system administrator, I want environment variables and secrets to be properly configured, so that the deployed application has access to required API keys and database credentials.

#### Acceptance Criteria

1. WHEN the application deploys, THE System SHALL preserve all existing environment variables (OPENAI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY, etc.)
2. WHEN the application deploys, THE System SHALL maintain the same secret management approach as the current deployment
3. WHEN the application starts, THE System SHALL have access to all required credentials

### Requirement 6: Backward Compatibility

**User Story:** As a developer, I want the new deployment method to maintain all existing functionality, so that the application behaves identically after migration.

#### Acceptance Criteria

1. WHEN the application deploys via source without build, THE System SHALL run the same Streamlit application
2. WHEN the application runs, THE System SHALL maintain the same port (8080) and configuration
3. WHEN the application runs, THE System SHALL maintain the same OAuth redirect URLs
4. WHEN the application runs, THE System SHALL maintain the same database connections and API integrations

### Requirement 7: Deployment Performance

**User Story:** As a developer, I want deployment times to be significantly faster, so that I can iterate more quickly.

#### Acceptance Criteria

1. WHEN deploying via source without build, THE System SHALL complete deployment faster than the current Cloud Build approach
2. WHEN comparing deployments, THE System SHALL measure and document the time savings

### Requirement 8: Rollback and Versioning

**User Story:** As a system administrator, I want to be able to rollback to previous versions if needed, so that I can quickly recover from deployment issues.

#### Acceptance Criteria

1. WHEN a deployment fails, THE System SHALL preserve the previous working version
2. WHEN needed, THE System SHALL allow rollback to a previous deployment
3. WHEN deploying, THE System SHALL maintain version history for troubleshooting
