# Requirements Document

## Introduction

The Self-Improving Geometry Extractor is a comprehensive system that transforms the current legal description processing application from a hardcoded regex + GPT approach into a self-improving, configuration-driven system. The system will normalize output into strict JSON schemas, externalize regex rules to Google Cloud Storage, implement automated testing harnesses, and create feedback loops that allow GPT to improve regex patterns based on failures. This evolution will enable the system to handle both explicit and abstract bearings while maintaining deterministic behavior and providing instant rollback capabilities.

## Glossary

- **LDR_System**: The Legal Description Reader application (Cloud Run + Streamlit + GCS)
- **Geometry_Extractor**: The core component that processes legal descriptions and extracts geometric data
- **Bucket_Classification**: The categorization of legal descriptions (explicit_bearings, abstract_bearings, no_bearings)
- **Gold_Dataset**: Manually verified correct outputs used as ground truth for testing
- **Harness**: The automated testing system that compares extractor output against gold dataset
- **Fingerprint**: A normalized representation of extracted line data for comparison
- **GCS_Rules**: Google Cloud Storage bucket containing versioned regex rule configurations
- **Patch_Generator**: GPT-powered system that creates regex rule modifications based on failures

## Requirements

### Requirement 1

**User Story:** As a system architect, I want normalized output schemas, so that all extracted data follows a consistent structure regardless of input type.

#### Acceptance Criteria

1. WHEN the system processes any legal description THEN the LDR_System SHALL output data in a strict JSON schema with bucket and lines array
2. WHEN a line is extracted THEN the LDR_System SHALL classify it as one of course, curve, ref_segment, or ref_curve types
3. WHEN line data is incomplete THEN the LDR_System SHALL use null values for missing fields rather than omitting them
4. WHEN bucket classification occurs THEN the LDR_System SHALL derive bucket after extraction based on line content
5. WHEN numeric course or curve lines exist THEN the LDR_System SHALL classify as explicit_bearings bucket

### Requirement 2

**User Story:** As a developer, I want externalized regex rules, so that pattern definitions can be modified without code deployment.

#### Acceptance Criteria

1. WHEN regex rules are needed THEN the LDR_System SHALL load patterns from rules.json configuration file
2. WHEN a rule is defined THEN the LDR_System SHALL include extractor_id, type, regex, and map properties
3. WHEN processing legal descriptions THEN the LDR_System SHALL loop through rules without hardcoded logic
4. WHEN rules are externalized THEN the LDR_System SHALL maintain identical behavior to current implementation
5. WHEN rule changes occur THEN the LDR_System SHALL not require code redeployment

### Requirement 3

**User Story:** As a system administrator, I want cloud-based rule storage, so that rules can be versioned and managed centrally.

#### Acceptance Criteria

1. WHEN GCS_Rules bucket is accessed THEN the LDR_System SHALL create bucket if it does not exist
2. WHEN rules are stored THEN the LDR_System SHALL enable object versioning on GCS_Rules bucket
3. WHEN a new rule version is created THEN the LDR_System SHALL save to rules/versions/rules_v{6-digit-number}.json
4. WHEN rules are deployed THEN the LDR_System SHALL copy versioned rules to rules/current.json
5. WHEN Cloud Run accesses rules THEN the LDR_System SHALL use service account with storage.objects.get permission

### Requirement 4

**User Story:** As a system operator, I want cached rule loading, so that the application remains responsive and resilient.

#### Acceptance Criteria

1. WHEN the application starts THEN the LDR_System SHALL load rules from GCS_Rules into memory cache
2. WHEN cache refresh occurs THEN the LDR_System SHALL reload rules every 300 seconds
3. WHEN rule compilation fails THEN the LDR_System SHALL retain previous working rules in memory
4. WHEN GCS_Rules is unavailable THEN the LDR_System SHALL continue operating with cached rules
5. WHEN regex compilation errors occur THEN the LDR_System SHALL never crash due to invalid rules

### Requirement 5

**User Story:** As a quality assurance engineer, I want a gold dataset, so that system accuracy can be measured and maintained.

#### Acceptance Criteria

1. WHEN creating gold dataset THEN the LDR_System SHALL collect approximately 40 real legal descriptions
2. WHEN processing descriptions for gold THEN the LDR_System SHALL save both original output and manually corrected output
3. WHEN storing gold data THEN the LDR_System SHALL use identical schema as production output
4. WHEN bad outputs exist THEN the LDR_System SHALL preserve incorrect outputs as they provide highest value for improvement
5. WHEN gold entries are created THEN the LDR_System SHALL store id, text, and gold_output fields

### Requirement 6

**User Story:** As a test engineer, I want fingerprinting capability, so that extracted lines can be compared for equivalence.

#### Acceptance Criteria

1. WHEN generating fingerprints THEN the LDR_System SHALL create one normalized fingerprint per extracted line
2. WHEN normalizing values THEN the LDR_System SHALL standardize case and round numeric values
3. WHEN fingerprinting courses THEN the LDR_System SHALL format as "course|S|45|12|30|E|125.00"
4. WHEN fingerprinting references THEN the LDR_System SHALL format as "ref_segment|row_line|duncan drive|northerly"
5. WHEN creating fingerprints THEN the LDR_System SHALL exclude raw text from fingerprint content

### Requirement 7

**User Story:** As a system validator, I want automated testing harness, so that extraction accuracy can be continuously monitored.

#### Acceptance Criteria

1. WHEN running harness THEN the LDR_System SHALL compare regex output against Gold_Dataset using fingerprints
2. WHEN differences are found THEN the LDR_System SHALL flag missing lines, extra lines, and bucket mismatches
3. WHEN harness completes THEN the LDR_System SHALL output failures.json with detailed failure information
4. WHEN testing gold cases THEN the LDR_System SHALL run harness against all Gold_Dataset entries
5. WHEN failures occur THEN the LDR_System SHALL provide sufficient detail for automated patch generation

### Requirement 8

**User Story:** As a system analyst, I want failure clustering, so that the most impactful issues can be prioritized.

#### Acceptance Criteria

1. WHEN analyzing failures THEN the LDR_System SHALL group failures by missing fingerprint patterns
2. WHEN clustering occurs THEN the LDR_System SHALL group failures by extractor_id for targeted fixes
3. WHEN prioritizing fixes THEN the LDR_System SHALL identify top 3 failure classes by frequency
4. WHEN processing one-off failures THEN the LDR_System SHALL ignore single-occurrence failures initially
5. WHEN clusters are identified THEN the LDR_System SHALL provide sufficient data for patch generation

### Requirement 9

**User Story:** As an AI system, I want GPT-powered patch generation, so that regex rules can be automatically improved.

#### Acceptance Criteria

1. WHEN generating patches THEN the Patch_Generator SHALL receive failing texts, expected fingerprints, and current regex rules
2. WHEN creating patches THEN the Patch_Generator SHALL output JSON with add_pattern, modify_pattern, and disable_pattern operations
3. WHEN patch format is specified THEN the Patch_Generator SHALL provide only JSON output without prose explanations
4. WHEN patches are generated THEN the Patch_Generator SHALL target specific failure patterns identified by clustering
5. WHEN patch operations are defined THEN the Patch_Generator SHALL ensure operations are atomic and reversible

### Requirement 10

**User Story:** As a system maintainer, I want automated patch application, so that improvements can be deployed without manual intervention.

#### Acceptance Criteria

1. WHEN applying patches THEN the LDR_System SHALL create new rules object from patch operations
2. WHEN validating patches THEN the LDR_System SHALL perform regex compilation check before deployment
3. WHEN compilation fails THEN the LDR_System SHALL discard invalid patches automatically
4. WHEN patches are valid THEN the LDR_System SHALL save as rules_v{incremented-number}.json in GCS_Rules versions
5. WHEN deployment occurs THEN the LDR_System SHALL copy new version to current.json for automatic reload

### Requirement 11

**User Story:** As a quality controller, I want regression testing, so that improvements don't introduce new failures.

#### Acceptance Criteria

1. WHEN patches are applied THEN the LDR_System SHALL re-run Harness against Gold_Dataset
2. WHEN comparing results THEN the LDR_System SHALL confirm failure count decreases from previous version
3. WHEN new failures appear THEN the LDR_System SHALL ensure no previously passing cases now fail
4. WHEN regression is detected THEN the LDR_System SHALL rollback to previous rules version automatically
5. WHEN improvements are confirmed THEN the LDR_System SHALL retain new rules version as current

### Requirement 12

**User Story:** As a machine learning engineer, I want GPT fine-tuning integration, so that classification accuracy improves over time.

#### Acceptance Criteria

1. WHEN storing training data THEN the LDR_System SHALL save raw text and final Bucket_Classification for each description
2. WHEN exporting training data THEN the LDR_System SHALL format as JSONL with messages array containing user and assistant entries
3. WHEN fine-tuning occurs THEN the LDR_System SHALL upload JSONL to OpenAI fine-tuning service
4. WHEN new models are ready THEN the LDR_System SHALL deploy updated classifier with new model ID
5. WHEN classification improves THEN the LDR_System SHALL maintain backward compatibility with existing data

### Requirement 13

**User Story:** As an end user, I want optimized runtime flow, so that processing is efficient and accurate.

#### Acceptance Criteria

1. WHEN processing input THEN the LDR_System SHALL run classifier first to determine processing path
2. WHEN no_bearings classification occurs THEN the LDR_System SHALL stop processing immediately
3. WHEN bearings are expected THEN the LDR_System SHALL run Geometry_Extractor with current rules
4. WHEN extraction is partial THEN the LDR_System SHALL use GPT candidate assistance for completion
5. WHEN output is generated THEN the LDR_System SHALL provide drawable lines in normalized format

### Requirement 14

**User Story:** As a domain expert, I want human override capability, so that incorrect classifications can be corrected and fed back into the system.

#### Acceptance Criteria

1. WHEN viewing results THEN the LDR_System SHALL display UI buttons for Explicit, Abstract, and None classifications
2. WHEN user clicks override THEN the LDR_System SHALL overwrite Gold_Dataset entry with corrected classification
3. WHEN corrections are made THEN the LDR_System SHALL store correction in database for future training
4. WHEN feedback is provided THEN the LDR_System SHALL feed corrections into Harness testing
5. WHEN training data is updated THEN the LDR_System SHALL include corrections in fine-tuning and regex patch processes

### Requirement 15

**User Story:** As a system administrator, I want maintenance workflows, so that the system continues to improve over time.

#### Acceptance Criteria

1. WHEN weekly maintenance occurs THEN the LDR_System SHALL add 20-50 new cases to Gold_Dataset
2. WHEN monthly maintenance occurs THEN the LDR_System SHALL retrain classifier with accumulated data
3. WHEN periodic maintenance occurs THEN the LDR_System SHALL prune dead regex patterns that no longer match
4. WHEN versioning maintenance occurs THEN the LDR_System SHALL maintain all historical versions without deletion
5. WHEN Gold_Dataset maintenance occurs THEN the LDR_System SHALL never delete gold entries as they provide ongoing value