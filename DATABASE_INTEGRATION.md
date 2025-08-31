# Database Integration for Classification Data

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
