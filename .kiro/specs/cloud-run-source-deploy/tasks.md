# Implementation Plan: Cloud Run Deploy from Source Without Build

## Overview

Migrate the LDR application from Cloud Build container image building to Cloud Run's "deploy from source without build" feature. This involves creating a source archive with dependencies, uploading to Cloud Storage, and deploying via gcloud with the --no-build flag. The GitHub Actions workflow will be updated to orchestrate this new deployment process.

## Tasks

- [x] 1. Create archive creation script
  - Write Python script to create .tar.gz archive with source and dependencies
  - Install requirements.txt to vendor directory
  - Exclude unnecessary files (.git, __pycache__, .env, .vscode, .cursor, etc.)
  - Verify archive size is under 250 MiB
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 1.1 Write property test for archive completeness
  - **Property 1: Archive Contains All Source Files**
  - **Validates: Requirements 1.1, 1.2, 1.5**

- [x] 1.2 Write property test for file exclusion
  - **Property 2: Archive Excludes Unnecessary Files**
  - **Validates: Requirements 1.3**

- [x] 1.3 Write property test for archive size
  - **Property 3: Archive Size Under Limit**
  - **Validates: Requirements 1.4**

- [x] 2. Create Cloud Storage upload script
  - Write Python script to authenticate with GCP
  - Upload archive to designated Cloud Storage bucket
  - Verify upload completion
  - Handle upload failures with retry logic
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 2.1 Write property test for upload success
  - **Property 4: Archive Upload Succeeds**
  - **Validates: Requirements 2.1, 2.2**

- [x] 3. Create Cloud Run deployment script
  - Write Python script to execute gcloud deploy command
  - Use --no-build flag with python311 base image
  - Set PORT=8080 and startup command (streamlit run main.py)
  - Pass all environment variables to Cloud Run
  - Verify service is running and accessible
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3.1 Write property test for deployment command correctness
  - **Property 5: Deployment Command Correctness**
  - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

- [x] 3.2 Write property test for service accessibility
  - **Property 6: Deployed Service Accessibility**
  - **Validates: Requirements 3.5**

- [x] 4. Update GitHub Actions workflow
  - Modify .github/workflows/deploy.yml to use new deployment process
  - Add steps for archive creation, upload, and deployment
  - Preserve existing environment variable handling
  - Add error reporting and logging
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 4.1 Write property test for workflow automation
  - **Property 10: Workflow Automation**
  - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**

- [x] 5. Verify environment variable preservation
  - Test that all existing environment variables are passed to Cloud Run
  - Verify OPENAI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY, etc. are accessible
  - Verify secrets are handled correctly
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 5.1 Write property test for environment variable preservation
  - **Property 7: Environment Variables Preserved**
  - **Validates: Requirements 5.1, 5.2, 5.3**

- [x] 6. Test functional equivalence
  - Deploy to dev branch and verify Streamlit app works identically
  - Test OAuth flow end-to-end
  - Verify database connections work
  - Verify API integrations work
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 6.1 Write property test for functional equivalence
  - **Property 8: Functional Equivalence After Migration**
  - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

- [x] 7. Checkpoint - Verify all core functionality works
  - Deploy to dev and test all features
  - Verify no regressions from current deployment
  - Ask the user if questions arise

- [x] 8. Measure deployment performance
  - Compare deployment time: source without build vs Cloud Build
  - Document time savings
  - _Requirements: 7.1, 7.2_

- [x] 8.1 Write property test for deployment performance
  - **Property 11: Deployment Performance Improvement**
  - **Validates: Requirements 7.1, 7.2**

- [x] 9. Implement rollback capability
  - Document rollback procedure
  - Test rollback to previous version
  - Verify version history is maintained
  - _Requirements: 8.1, 8.2, 8.3_

- [x] 9.1 Write property test for rollback capability
  - **Property 12: Rollback Capability**
  - **Validates: Requirements 8.1, 8.2, 8.3**

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise

- [x] 11. Deploy to production
  - Deploy to main branch using new process
  - Monitor Cloud Run logs for any issues
  - Verify service is running and accessible
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- All scripts should be idempotent and handle errors gracefully
