# Database Integration for Classification Data

## Update 2025-08-31T20:13:32Z - Enhanced Data Collection (WARP USAGE 1104)

All possible data fields numbered:

Processing metrics:
1. processing_time - How long GPT analysis took
2. page_count - Number of PDF pages processed
3. text_length - Characters extracted via OCR
4. bearing_count - Number of bearings found
5. parsed_bearing_count - Number successfully parsed to coordinates

Quality/confidence metrics:
6. parsing_success_rate - % of bearings that parsed correctly
7. ocr_confidence - How clear the PDF text was (if available)
8. model_version - Which GPT model used ("ft:gpt-3.5-turbo-0125:personal:ldr")
9. supplemental_info_found - Boolean: found Land Lot/County data?

File metadata:
10. file_size - PDF size in bytes
11. file_hash - MD5/SHA256 for duplicate detection
12. original_filename - User's original filename
13. storage_path - Where file is stored (if we store files)

User/session context:
14. upload_method - "file_upload", "local_file", "google_drive"
15. session_id - Track multiple docs in same browser session
16. debug_mode - Was debug mode enabled?
17. ip_address - User's IP (if needed for analytics)
18. user_agent - Browser info

Processing context:
19. prompt_version - Which prompt template was used
20. temperature - GPT temperature setting (0.1)
21. max_tokens - If we set token limits

Which numbers do you want to add?
C:\dev\ldr.ai
i mean i like them all but we already have 12 and im not sure how you'll do 7 but it sounds good
You're right! We already have #12 (original_filename/filename) and user_email.

For #7 (OCR confidence): Tesseract actually gives confidence scores! We can get per-word confidence and average it:
python
So let's add the easy ones first:

High priority (easy to implement):
•  1, 2, 3, 4, 5 (processing metrics)
•  6 (parsing success rate) 
•  8, 9 (model version, supplemental info found)
•  10, 11 (file size, file hash)
•  14, 16 (upload method, debug mode)

Medium priority:
•  7 (OCR confidence - requires tesseract changes)
•  19, 20 (prompt version, temperature)

Maybe later:
•  15, 17, 18 (session/IP/user agent)

Want to start with the high priority ones? That's about 10 new fields that are straightforward to collect.

### New Database Columns Added
Expanded the `classification_data` table with additional generated columns for comprehensive analytics and machine learning training data:

```sql
-- User and file tracking columns (already added)
ALTER TABLE classification_data ADD COLUMN filename TEXT GENERATED ALWAYS AS (reasoning_data->>'filename') STORED;
ALTER TABLE classification_data ADD COLUMN user_email TEXT GENERATED ALWAYS AS (reasoning_data->>'user_email') STORED;

-- Processing metrics
ALTER TABLE classification_data ADD COLUMN processing_time NUMERIC GENERATED ALWAYS AS ((reasoning_data->>'processing_time')::numeric) STORED;
ALTER TABLE classification_data ADD COLUMN page_count INTEGER GENERATED ALWAYS AS ((reasoning_data->>'page_count')::integer) STORED;
ALTER TABLE classification_data ADD COLUMN text_length INTEGER GENERATED ALWAYS AS ((reasoning_data->>'text_length')::integer) STORED;
ALTER TABLE classification_data ADD COLUMN bearing_count INTEGER GENERATED ALWAYS AS ((reasoning_data->>'bearing_count')::integer) STORED;
ALTER TABLE classification_data ADD COLUMN parsed_bearing_count INTEGER GENERATED ALWAYS AS ((reasoning_data->>'parsed_bearing_count')::integer) STORED;

-- Quality metrics
ALTER TABLE classification_data ADD COLUMN parsing_success_rate NUMERIC GENERATED ALWAYS AS ((reasoning_data->>'parsing_success_rate')::numeric) STORED;
ALTER TABLE classification_data ADD COLUMN ocr_confidence NUMERIC GENERATED ALWAYS AS ((reasoning_data->>'ocr_confidence')::numeric) STORED;
ALTER TABLE classification_data ADD COLUMN model_version TEXT GENERATED ALWAYS AS (reasoning_data->>'model_version') STORED;
ALTER TABLE classification_data ADD COLUMN supplemental_info_found BOOLEAN GENERATED ALWAYS AS ((reasoning_data->>'supplemental_info_found')::boolean) STORED;

-- File metadata  
ALTER TABLE classification_data ADD COLUMN file_size BIGINT GENERATED ALWAYS AS ((reasoning_data->>'file_size')::bigint) STORED;
ALTER TABLE classification_data ADD COLUMN file_hash TEXT GENERATED ALWAYS AS (reasoning_data->>'file_hash') STORED;
ALTER TABLE classification_data ADD COLUMN storage_path TEXT GENERATED ALWAYS AS (reasoning_data->>'storage_path') STORED;

-- Context tracking
ALTER TABLE classification_data ADD COLUMN upload_method TEXT GENERATED ALWAYS AS (reasoning_data->>'upload_method') STORED;
ALTER TABLE classification_data ADD COLUMN debug_mode BOOLEAN GENERATED ALWAYS AS ((reasoning_data->>'debug_mode')::boolean) STORED;
ALTER TABLE classification_data ADD COLUMN prompt_version TEXT GENERATED ALWAYS AS (reasoning_data->>'prompt_version') STORED;
ALTER TABLE classification_data ADD COLUMN temperature NUMERIC GENERATED ALWAYS AS ((reasoning_data->>'temperature')::numeric) STORED;

-- Add corresponding indexes for performance
CREATE INDEX idx_classification_data_filename ON classification_data(filename);
CREATE INDEX idx_classification_data_user_email ON classification_data(user_email);
CREATE INDEX idx_classification_data_file_hash ON classification_data(file_hash);
CREATE INDEX idx_classification_data_upload_method ON classification_data(upload_method);
CREATE INDEX idx_classification_data_processing_time ON classification_data(processing_time);
CREATE INDEX idx_classification_data_bearing_count ON classification_data(bearing_count);
```

### Enhanced Data Structure
The reasoning_data JSON now includes comprehensive metadata:

```json
{
  "filename": "property_survey.pdf",
  "user_email": "user@example.com",
  "timestamp": "2025-08-31T20:13:32Z",
  "processing_time": 2.45,
  "page_count": 3,
  "text_length": 1247,
  "bearing_count": 8,
  "parsed_bearing_count": 7,
  "parsing_success_rate": 87.5,
  "ocr_confidence": 94.2,
  "model_version": "ft:gpt-3.5-turbo-0125:personal:ldr",
  "supplemental_info_found": true,
  "file_size": 245760,
  "file_hash": "a1b2c3d4e5f6...",
  "storage_path": "pdfs/user123/20250831_property_survey.pdf",
  "upload_method": "file_upload",
  "debug_mode": false,
  "prompt_version": "v2.1",
  "temperature": 0.1,
  "input_text": "Legal description text...",
  "classification": "explicit_bearings",
  "confidence": "high",
  "reasoning": "AI explanation...",
  "evidence": "Evidence found...",
  "alternatives": "Alternative classifications...",
  "full_response": "Complete GPT response..."
}
```

### New Analytics Capabilities

**Performance Analytics:**
- Track processing times across different file sizes
- Monitor GPT model performance and success rates
- Identify optimal document characteristics

**User Analytics:**
- Track user activity and document processing patterns
- Monitor upload method preferences
- Analyze user success rates with different document types

**File Management:**
- Deduplicate files using hash comparison
- Track storage usage and optimize costs
- Link multiple processing attempts of same document

**Quality Metrics:**
- OCR confidence vs parsing success correlation
- Model version performance comparison
- Supplemental information extraction success rates

### Machine Learning Training Data
All collected metrics provide rich training data for:
- Document quality prediction models
- Processing time estimation
- Success rate prediction based on file characteristics
- User experience optimization

---

## Overview
This update migrates classification reasoning data from local JSON file storage to Supabase PostgreSQL database for better persistence, scalability, and multi-user support.

## Changes Made

### 1. Database Schema (`classification_table.sql`)
- Created `classification_data` table with JSONB storage
- Stores complete classification reasoning data as JSON
- Generated columns for fast filtering (classification, confidence, timestamp)
- Proper indexing for performance
- Row Level Security policies

### 2. Database Functions (`utils/classification.py`)
- `save_classification_data()` - Save classification entries to database
- `load_all_classification_data()` - Load all classification data
- `get_filtered_classification_data()` - Load filtered data with performance limits
- `clear_all_classification_data()` - Admin function to clear all data
- `get_classification_stats()` - Get summary statistics
- `test_database_connection()` - Connection testing

### 3. Main Application Updates (`main.py`)
- Updated `extract_bearings_with_gpt()` function
- Changed from JSON file writing to database saving
- Uses `save_classification_data()` instead of file append
- Maintains all existing functionality

### 4. Reasoning Page Updates (`pages/reasoning.py`)
- Completely migrated from JSON file reading to database queries
- Added database connection testing
- Improved filtering using database queries
- Added admin clear functionality with password protection ('warez')
- Maintained all existing UI and export functionality
- Better error handling and user feedback

## Data Structure
The database stores the same JSON structure as before:
```json
{
  "timestamp": "2025-08-31T05:45:00Z",
  "input_text": "Legal description text...",
  "classification": "explicit_bearings",
  "confidence": "high", 
  "reasoning": "AI explanation...",
  "evidence": "Evidence found...",
  "alternatives": "Alternative classifications...",
  "full_response": "Complete GPT response..."
}
```

## Key Features

### Shared Knowledge Base
- All classification data is globally accessible (no user restrictions)
- Everyone can learn from all classifications
- Better analytics with more data points

### Performance
- Database indexing for fast queries
- Filtered loading with limits (50 records max)
- Efficient JSON storage with PostgreSQL JSONB

### Admin Controls
- Password-protected database clearing ('warez')
- Connection testing and error handling
- Debug mode integration

### Export Functionality
- JSON export (same as before)
- CSV export with structured data
- Filtered exports based on current view

## Migration Notes
- No automatic migration from existing JSON files
- Fresh start with database storage
- Existing JSON files remain but are not used
- All new classifications save to database

## Database Requirements
- Supabase project with PostgreSQL
- Run `classification_table.sql` to create the table
- Proper environment variables for Supabase connection
- Internet connectivity for database access

## Backwards Compatibility
- All existing UI functionality maintained
- Same export formats available
- Debug mode still works
- No breaking changes to user experience
