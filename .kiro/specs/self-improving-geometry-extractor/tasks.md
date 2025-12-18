# Implementation Plan

- [x] 1. Normalize Output Schema (Foundation)


  - Create strict JSON schema classes for bucket and lines array structure
  - Implement schema validation and null value handling for missing fields
  - Modify existing GPT extraction to output normalized schema
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 1.1 Write property test for schema consistency


  - **Property 1: Schema consistency across all outputs**
  - **Validates: Requirements 1.1**

- [x] 1.2 Write property test for line type classification

  - **Property 2: Line type classification completeness**
  - **Validates: Requirements 1.2**

- [x] 1.3 Write property test for null value consistency

  - **Property 3: Null value consistency for missing data**
  - **Validates: Requirements 1.3**

- [x] 1.4 Implement bucket classification logic


  - Create bucket derivation function that analyzes extracted line content
  - Implement explicit_bearings detection for numeric course/curve lines
  - Update existing classification flow to use post-extraction bucket determination
  - _Requirements: 1.4, 1.5_

- [x] 1.5 Write property test for bucket derivation

  - **Property 4: Bucket derivation from extraction results**
  - **Validates: Requirements 1.4**

- [x] 1.6 Write property test for explicit bearings classification

  - **Property 5: Explicit bearings bucket classification**
  - **Validates: Requirements 1.5**

- [x] 2. Externalize Regex Rules


  - Extract current hardcoded regex patterns from main.py into rules.json structure
  - Create RulesConfig class with extractor_id, type, regex, and map properties
  - Implement rule loading and processing loop without hardcoded logic
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 2.1 Write property test for behavioral preservation


  - **Property 6: Behavioral preservation during externalization**
  - **Validates: Requirements 2.4**

- [x] 2.2 Create rules.json configuration file

  - Define JSON schema for regex rules with required properties
  - Convert existing regex patterns to configuration format
  - Implement rule validation and compilation checking
  - _Requirements: 2.2, 2.4, 2.5_

- [x] 3. Implement GCS Rules Infrastructure


  - Create GCS bucket with versioning enabled for rule storage
  - Implement GCSRulesManager class for version management
  - Create rules directory structure: current.json and versions/
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 3.1 Set up GCS bucket and permissions


  - Create ldr-rules-bucket with object versioning enabled
  - Configure Cloud Run service account with storage.objects.get permission
  - Implement bucket creation and initial directory structure
  - _Requirements: 3.1, 3.2, 3.5_

- [x] 3.2 Implement rule versioning system


  - Create version numbering with 6-digit format (rules_v000001.json)
  - Implement save_rules_version and rollback_to_version methods
  - Add version tracking in rule_versions database table
  - _Requirements: 3.3, 3.4_

- [x] 4. Create Cached Rule Loader



  - Implement memory cache for rules with 300-second refresh cycle
  - Add regex compilation validation and error handling
  - Create fallback mechanism for GCS unavailability
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 4.1 Write property test for rule resilience


  - **Property 7: Resilience to invalid regex rules**
  - **Validates: Requirements 4.3**

- [x] 4.2 Write property test for crash prevention

  - **Property 8: Crash prevention from rule errors**
  - **Validates: Requirements 4.5**

- [x] 4.3 Implement graceful error handling

  - Add try-catch blocks for regex compilation failures
  - Implement last-known-good rules retention
  - Create logging for rule loading issues and fallback scenarios
  - _Requirements: 4.5_

- [ ] 5. Create Gold Dataset Infrastructure
  - Design gold_dataset database table schema
  - Implement GoldDatasetManager class for CRUD operations
  - Collect initial 40 real legal descriptions for gold dataset
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 5.1 Write property test for gold dataset completeness
  - **Property 9: Gold dataset completeness**
  - **Validates: Requirements 5.2**

- [ ] 5.2 Implement gold dataset collection workflow
  - Create UI for manual verification and correction of outputs
  - Implement storage of both original and corrected outputs
  - Add gold dataset entry creation with required fields (id, text, gold_output)
  - _Requirements: 5.2, 5.5_

- [ ] 6. Implement Fingerprinting System
  - Create fingerprint generation algorithm for normalized line comparison
  - Implement case standardization and numeric value rounding
  - Create type-specific fingerprint formats for course, curve, and reference lines
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 6.1 Write property test for fingerprint correspondence
  - **Property 10: Fingerprint-to-line correspondence**
  - **Validates: Requirements 6.1**

- [ ] 6.2 Write property test for fingerprint normalization
  - **Property 11: Fingerprint normalization consistency**
  - **Validates: Requirements 6.2**

- [ ] 6.3 Implement fingerprint format specifications
  - Create course fingerprint format: "course|S|45|12|30|E|125.00"
  - Create reference fingerprint format: "ref_segment|row_line|duncan drive|northerly"
  - Exclude raw text from fingerprint content
  - _Requirements: 6.3, 6.4, 6.5_

- [ ] 7. Build Testing Harness
  - Create automated comparison system using fingerprints
  - Implement failure detection for missing, extra, and mismatched lines
  - Generate detailed failures.json output for patch generation
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 7.1 Write property test for harness comparison method
  - **Property 12: Harness fingerprint-based comparison**
  - **Validates: Requirements 7.1**

- [ ] 7.2 Implement harness execution engine
  - Create test runner that processes all gold dataset entries
  - Implement fingerprint-based comparison logic
  - Generate structured failure reports with sufficient detail for automation
  - _Requirements: 7.2, 7.3, 7.4_

- [ ] 8. Create Failure Clustering System
  - Implement clustering by missing fingerprint patterns
  - Add clustering by extractor_id for targeted fixes
  - Create prioritization logic to identify top 3 failure classes
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 8.1 Implement clustering algorithms
  - Group failures by missing fingerprint patterns
  - Group failures by extractor_id that failed to match
  - Filter out single-occurrence failures for initial focus
  - _Requirements: 8.1, 8.2, 8.4_

- [ ] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Build GPT Patch Generator
  - Create prompt template for patch generation based on failure analysis
  - Implement patch format with add_pattern, modify_pattern, disable_pattern operations
  - Ensure JSON-only output without prose explanations
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ] 10.1 Design patch generation prompt
  - Create structured prompt that includes failing texts, expected fingerprints, and current rules
  - Specify JSON output format with required operation types
  - Target specific failure patterns identified by clustering
  - _Requirements: 9.1, 9.4_

- [ ] 11. Implement Patch Application System
  - Create patch validation with regex compilation checking
  - Implement automatic patch discard for compilation failures
  - Add versioned rule saving and current.json update mechanism
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 11.1 Write property test for patch validation
  - **Property 13: Patch validation before deployment**
  - **Validates: Requirements 10.2**

- [ ] 11.2 Implement patch application workflow
  - Apply patch operations to create new rules object
  - Validate regex compilation before deployment
  - Save valid patches as incremented version files in GCS
  - _Requirements: 10.1, 10.4, 10.5_

- [ ] 12. Create Regression Testing System
  - Implement automatic harness re-run after patch application
  - Add failure count comparison and improvement validation
  - Create automatic rollback mechanism for regressions
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [ ] 12.1 Write property test for improvement validation
  - **Property 14: Improvement validation through failure reduction**
  - **Validates: Requirements 11.2**

- [ ] 12.2 Write property test for automatic rollback
  - **Property 15: Automatic rollback on regression**
  - **Validates: Requirements 11.4**

- [ ] 12.3 Implement regression detection and rollback
  - Compare new version failure count with previous version
  - Detect when previously passing cases now fail
  - Automatically rollback to previous version when regression detected
  - _Requirements: 11.3, 11.4_

- [ ] 13. Integrate GPT Fine-Tuning Loop
  - Implement training data collection with raw text and final bucket classification
  - Create JSONL export format for OpenAI fine-tuning
  - Add fine-tuning upload and model deployment workflow
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 13.1 Write property test for training data format
  - **Property 16: Training data export format compliance**
  - **Validates: Requirements 12.2**

- [ ] 13.2 Implement fine-tuning data pipeline
  - Store raw text and bucket classification for each processed description
  - Export data in JSONL format with proper message structure
  - Maintain backward compatibility with existing data
  - _Requirements: 12.1, 12.5_

- [ ] 14. Optimize Runtime Processing Flow
  - Implement classifier-first processing order
  - Add early termination for no_bearings classification
  - Integrate regex extractor with GPT candidate assistance for partial results
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

- [ ] 14.1 Write property test for processing order
  - **Property 17: Classification-first processing order**
  - **Validates: Requirements 13.1**

- [ ] 14.2 Write property test for early termination
  - **Property 18: Early termination for no-bearings classification**
  - **Validates: Requirements 13.2**

- [ ] 14.3 Implement optimized processing pipeline
  - Run classifier before extraction processing
  - Stop processing immediately for no_bearings classification
  - Use GPT assistance for partial extraction results
  - _Requirements: 13.2, 13.4_

- [ ] 15. Add Human Override Interface
  - Create UI buttons for Explicit, Abstract, and None classification overrides
  - Implement gold dataset correction workflow
  - Integrate corrections into harness testing and training data
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

- [ ] 15.1 Design override user interface
  - Add classification override buttons to results display
  - Implement correction storage in database
  - Feed corrections into harness and fine-tuning processes
  - _Requirements: 14.1, 14.3, 14.5_

- [ ] 16. Implement Maintenance Workflows
  - Create periodic maintenance for gold dataset expansion
  - Add regex pattern pruning for unused rules
  - Implement version preservation without deletion
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

- [ ] 16.1 Create maintenance automation
  - Implement pruning of dead regex patterns that no longer match
  - Ensure all historical versions are preserved without deletion
  - Maintain gold dataset entries permanently for ongoing value
  - _Requirements: 15.3, 15.4, 15.5_

- [ ] 17. Final Integration and Testing
  - Integrate all components into unified self-improving system
  - Run complete end-to-end testing with real legal descriptions
  - Validate that system improves itself through automated feedback loops
  - _Requirements: All_

- [ ] 18. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.