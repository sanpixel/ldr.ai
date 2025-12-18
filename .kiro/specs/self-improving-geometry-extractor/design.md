# Design Document

## Overview

The Self-Improving Geometry Extractor transforms the current Legal Description Reader from a hardcoded regex + GPT system into a sophisticated, self-improving architecture. The system maintains the existing Streamlit + Cloud Run + Supabase foundation while adding Google Cloud Storage for rule management, automated testing harnesses, and GPT-powered improvement loops.

The core innovation is the separation of extraction logic from code deployment - regex patterns become configuration data stored in GCS, allowing the system to evolve its parsing capabilities through automated feedback loops without requiring code changes.

## Architecture

### Current State Analysis
- **Frontend**: Streamlit web application
- **Backend**: Cloud Run serverless containers  
- **Database**: Supabase PostgreSQL with `classification_data` table
- **AI**: OpenAI GPT-4 fine-tuned model for classification and extraction
- **Output**: Inconsistent data structures, hardcoded regex patterns
- **Testing**: Manual verification only

### Target Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Cloud Run      │    │   GCS Rules     │
│   Frontend      │◄──►│   Application    │◄──►│   Storage       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Supabase      │    │   Testing        │    │   Versioned     │
│   Database      │◄──►│   Harness        │    │   Rules         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   GPT Patch      │
                    │   Generator      │
                    └──────────────────┘
```

### Data Flow
1. **Input Processing**: Legal description → GPT Classifier → Bucket determination
2. **Rule Loading**: GCS Rules → Memory Cache → Regex Compilation
3. **Extraction**: Regex Rules → Structured Output → Fingerprinting
4. **Testing**: Gold Dataset → Harness → Failure Analysis
5. **Improvement**: Failure Clustering → GPT Patches → Rule Updates
6. **Deployment**: New Rules → GCS Versioning → Auto-reload

## Components and Interfaces

### 1. Schema Normalizer
**Purpose**: Enforce consistent output structure across all processing paths

**Input**: Raw extraction results from regex or GPT
**Output**: Normalized JSON schema

```python
class SchemaOutput:
    bucket: str  # explicit_bearings | abstract_bearings | no_bearings
    lines: List[LineData]

class LineData:
    type: str     # course | curve | ref_segment | ref_curve  
    idx: int      # sequence number
    raw: str      # original text
    # Type-specific fields (null if not applicable)
    cardinal_ns: Optional[str]
    degrees: Optional[int] 
    minutes: Optional[int]
    seconds: Optional[float]
    cardinal_ew: Optional[str]
    distance: Optional[float]
    monument: Optional[str]
    reference: Optional[str]
```

### 2. GCS Rules Manager
**Purpose**: Handle versioned storage and retrieval of regex rule configurations

**Interface**:
```python
class GCSRulesManager:
    def load_current_rules() -> RulesConfig
    def save_rules_version(rules: RulesConfig) -> str  # returns version_id
    def rollback_to_version(version_id: str) -> bool
    def list_versions() -> List[str]
```

**Rules Schema**:
```json
{
  "version": "000001",
  "rules": [
    {
      "extractor_id": "bearing_dms_standard", 
      "type": "course",
      "regex": "([NS])\\s*(\\d+)°\\s*(\\d+)'\\s*(\\d+)\"\\s*([EW])",
      "map": {
        "cardinal_ns": 1,
        "degrees": 2, 
        "minutes": 3,
        "seconds": 4,
        "cardinal_ew": 5
      }
    }
  ]
}
```

### 3. Cached Rule Loader
**Purpose**: Provide resilient, performant access to regex rules

**Features**:
- 300-second refresh cycle from GCS
- Regex compilation validation
- Fallback to last-known-good rules
- Graceful degradation when GCS unavailable

### 4. Gold Dataset Manager
**Purpose**: Maintain verified correct outputs for testing and training

**Schema**:
```python
class GoldEntry:
    id: str
    text: str           # original legal description
    gold_output: SchemaOutput  # manually verified correct result
    created_at: datetime
    verified_by: str    # user who verified
```

### 5. Fingerprinting Engine
**Purpose**: Create normalized representations for comparison

**Algorithm**:
```python
def generate_fingerprint(line: LineData) -> str:
    if line.type == "course":
        return f"course|{line.cardinal_ns[0]}|{line.degrees}|{line.minutes}|{line.seconds}|{line.cardinal_ew[0]}|{line.distance:.2f}"
    elif line.type == "ref_segment":
        return f"ref_segment|{normalize_text(line.reference)}|{normalize_text(line.monument)}"
    # ... other types
```

### 6. Testing Harness
**Purpose**: Automated comparison of extractor output against gold dataset

**Process**:
1. Run current extractor on all gold dataset texts
2. Generate fingerprints for both actual and expected outputs  
3. Compare fingerprint sets to identify missing/extra/mismatched lines
4. Generate detailed failure report with context for patch generation

**Output**: `failures.json` with structured failure data

### 7. Failure Clustering Engine
**Purpose**: Group failures by pattern to prioritize fixes

**Clustering Dimensions**:
- Missing fingerprint patterns (what should have been extracted)
- Extractor ID (which regex rule failed)
- Frequency (how often this failure occurs)

### 8. GPT Patch Generator
**Purpose**: Generate regex rule modifications based on failure analysis

**Input**:
- Failing text samples
- Expected fingerprints  
- Current regex rules
- Failure cluster analysis

**Output**: JSON patch operations
```json
{
  "patches": [
    {
      "operation": "modify_pattern",
      "extractor_id": "bearing_dms_standard",
      "new_regex": "([NS])\\s*(\\d+)\\s*[°degrees]\\s*(\\d+)\\s*[''minutes]",
      "reason": "Handle 'degrees' and 'minutes' spelled out"
    }
  ]
}
```

## Data Models

### Current Database Schema (Preserved)
```sql
-- Existing classification_data table remains unchanged
CREATE TABLE classification_data (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  reasoning_data JSONB NOT NULL,
  classification TEXT GENERATED ALWAYS AS (reasoning_data->>'classification') STORED,
  confidence TEXT GENERATED ALWAYS AS (reasoning_data->>'confidence') STORED,
  created_at TIMESTAMPTZ GENERATED ALWAYS AS ((reasoning_data->>'timestamp')::timestamptz) STORED,
  inserted_at TIMESTAMPTZ DEFAULT NOW()
);
```

### New Tables
```sql
-- Gold dataset for testing
CREATE TABLE gold_dataset (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  text TEXT NOT NULL,
  gold_output JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  verified_by TEXT,
  source_file TEXT
);

-- Rule version tracking
CREATE TABLE rule_versions (
  version_id TEXT PRIMARY KEY,
  gcs_path TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  performance_metrics JSONB,
  is_active BOOLEAN DEFAULT FALSE
);

-- Test results tracking  
CREATE TABLE harness_results (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  rule_version_id TEXT REFERENCES rule_versions(version_id),
  total_cases INTEGER,
  passed_cases INTEGER,
  failed_cases INTEGER,
  failure_details JSONB,
  run_at TIMESTAMPTZ DEFAULT NOW()
);
```

### GCS Storage Layout
```
gs://ldr-rules-bucket/
├── rules/
│   ├── current.json              # Active rules
│   └── versions/
│       ├── rules_v000001.json    # Version history
│       ├── rules_v000002.json
│       └── ...
├── gold-dataset/
│   ├── descriptions/             # Source texts
│   └── outputs/                  # Verified outputs
└── test-results/
    ├── harness-runs/            # Test execution logs
    └── failure-analysis/        # Clustered failure reports
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Property 1: Schema consistency across all outputs
*For any* legal description input, the system output should always contain a bucket field and lines array with valid structure
**Validates: Requirements 1.1**

Property 2: Line type classification completeness  
*For any* extracted line, the type field should always be one of: course, curve, ref_segment, or ref_curve
**Validates: Requirements 1.2**

Property 3: Null value consistency for missing data
*For any* line with incomplete data, missing fields should be null rather than omitted from the output structure
**Validates: Requirements 1.3**

Property 4: Bucket derivation from extraction results
*For any* legal description, bucket classification should be determined solely from extracted line content, not original text
**Validates: Requirements 1.4**

Property 5: Explicit bearings bucket classification
*For any* output containing numeric course or curve lines, the bucket should be classified as explicit_bearings
**Validates: Requirements 1.5**

Property 6: Behavioral preservation during externalization
*For any* legal description processed by both old and new systems, the outputs should be functionally equivalent
**Validates: Requirements 2.4**

Property 7: Resilience to invalid regex rules
*For any* invalid regex pattern introduced to the system, the application should continue operating with previous valid rules
**Validates: Requirements 4.3**

Property 8: Crash prevention from rule errors
*For any* regex compilation error, the system should handle the error gracefully without application termination
**Validates: Requirements 4.5**

Property 9: Gold dataset completeness
*For any* gold dataset entry, both original extractor output and manually corrected output should be preserved
**Validates: Requirements 5.2**

Property 10: Fingerprint-to-line correspondence
*For any* set of extracted lines, the number of generated fingerprints should equal the number of lines
**Validates: Requirements 6.1**

Property 11: Fingerprint normalization consistency
*For any* two equivalent line representations with different formatting, their fingerprints should be identical
**Validates: Requirements 6.2**

Property 12: Harness fingerprint-based comparison
*For any* harness test run, comparisons between actual and expected outputs should use fingerprints rather than raw text
**Validates: Requirements 7.1**

Property 13: Patch validation before deployment
*For any* generated patch, regex compilation validation should occur before the patch is applied to production rules
**Validates: Requirements 10.2**

Property 14: Improvement validation through failure reduction
*For any* new rule version, the failure count should be less than or equal to the previous version's failure count
**Validates: Requirements 11.2**

Property 15: Automatic rollback on regression
*For any* rule version that increases failure count, the system should automatically revert to the previous version
**Validates: Requirements 11.4**

Property 16: Training data export format compliance
*For any* exported training data, the format should be valid JSONL with proper message structure for OpenAI fine-tuning
**Validates: Requirements 12.2**

Property 17: Classification-first processing order
*For any* input legal description, classification should complete before extraction processing begins
**Validates: Requirements 13.1**

Property 18: Early termination for no-bearings classification
*For any* input classified as no_bearings, extraction processing should not be initiated
**Validates: Requirements 13.2**

<function_calls>
<invoke name="prework">
<parameter name="featureName">self-improving-geometry-extractor

## Error Handling

### Rule Loading Failures
- **GCS Unavailable**: Continue with cached rules, log warning, retry on next refresh cycle
- **Invalid JSON**: Reject new rules, retain current rules, alert administrators
- **Regex Compilation Errors**: Skip invalid patterns, log details, continue with valid patterns
- **Version Conflicts**: Use timestamp-based resolution, maintain audit trail

### Extraction Failures  
- **No Regex Matches**: Fall back to GPT extraction with current prompt
- **Partial Matches**: Combine regex results with GPT assistance for missing elements
- **Schema Validation Errors**: Apply default values, log validation failures
- **Timeout Errors**: Return partial results with timeout flag

### Testing Harness Failures
- **Gold Dataset Corruption**: Skip corrupted entries, log issues, continue with valid entries
- **Fingerprint Generation Errors**: Use raw text comparison as fallback
- **Comparison Timeouts**: Mark as inconclusive, retry with smaller batches
- **Storage Failures**: Cache results locally, retry upload when storage available

### Patch Application Failures
- **Invalid Patch Format**: Reject patch, log error, continue with current rules
- **Regex Compilation Failure**: Rollback to previous version automatically
- **GCS Upload Failure**: Retry with exponential backoff, maintain local backup
- **Version Increment Conflicts**: Use atomic operations with conflict resolution

## Testing Strategy

### Dual Testing Approach
The system requires both unit testing and property-based testing to ensure correctness:

**Unit Tests** verify specific examples, edge cases, and error conditions:
- Schema validation with known inputs and expected outputs
- Rule loading with various JSON configurations
- Fingerprint generation with specific line types
- Error handling with simulated failure conditions

**Property-Based Tests** verify universal properties across all inputs:
- Schema consistency across randomly generated legal descriptions
- Fingerprint normalization with equivalent but differently formatted data
- Rule resilience with randomly generated invalid regex patterns
- Improvement validation with synthetic failure scenarios

**Property-Based Testing Framework**: Use Hypothesis for Python to generate test cases and verify properties hold across 100+ iterations per test.

**Test Tagging**: Each property-based test must include a comment with the format:
`# Feature: self-improving-geometry-extractor, Property {number}: {property_text}`

### Integration Testing
- **End-to-End Workflows**: Complete processing pipelines from input to drawable output
- **GCS Integration**: Rule loading, versioning, and rollback scenarios  
- **Database Integration**: Gold dataset operations and harness result storage
- **GPT Integration**: Patch generation and fine-tuning data export

### Performance Testing
- **Rule Loading Performance**: Measure cache refresh times and memory usage
- **Extraction Performance**: Benchmark regex vs GPT processing times
- **Harness Performance**: Test scalability with large gold datasets
- **Storage Performance**: Measure GCS upload/download times for rule versions

### Regression Testing
- **Behavioral Preservation**: Ensure externalized rules produce identical results
- **Performance Regression**: Monitor processing times across rule versions
- **Accuracy Regression**: Track failure rates in harness results over time
- **API Compatibility**: Maintain backward compatibility for existing integrations