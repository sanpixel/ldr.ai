# LDR.ai Refactoring Plan

## Overview
This document outlines the refactoring needed for GCS PDF storage and GPT API calls to eliminate redundancy and improve consistency.

## Current Issues
- Multiple similar GCS upload functions doing overlapping tasks
- Inconsistent file naming patterns in GCS bucket
- GPT calls scattered across multiple functions
- Duplicate code patterns between `process_pdf()` and `process_image()`

---

## GCS Storage Functions Analysis

### Current Functions in `utils/gcs_storage.py`

| Function | Line | Purpose | Naming Pattern | Issues |
|----------|------|---------|----------------|---------|
| `upload_pdf_preview()` | 44 | Upload highlighted PDFs | UUID filenames | Good for versioning |
| `upload_pdf_file()` | 85 | Upload original PDFs | "init-" prefix | Redundant with upload_pdf_image |
| `upload_pdf_image()` | 125 | Upload PDF images | Customizable prefix | Most flexible |

### Current GCS Bucket Files

| File | Size | Created By | Line |
|------|------|------------|------|
| `highlighted-5 1994 COBB Book 16131 Page 2451.png` | 556.3 KB | Line 1474 or 2239 | `upload_pdf_image` with "highlighted" prefix |
| `highlighted_92aab43f-4d50-4981-b9d1-047d072c12a1.png` | 500.6 KB | Line 555 | `upload_pdf_preview` with UUID |
| `init-5 1994 COBB Book 16131 Page 2451.pdf` | 614.9 KB | Line 1382 | `upload_pdf_file` with "init-" prefix |
| `init-5 1994 COBB Book 16131 Page 2451.png` | 500.6 KB | Line 1409 or 2215 | `upload_pdf_image` with "init-" prefix |
| `init-example-gwinnett.png` | - | Line 805 | `upload_pdf_image` for example |

### GCS Upload Call Locations in `main.py`

| Line | Function Called | Context | Purpose |
|------|----------------|---------|---------|
| 555 | `upload_pdf_preview` | After GPT processing | Store highlighted preview with UUID |
| 797 | `upload_pdf_image` | Example PDF loading | Store example image |
| 805 | `upload_pdf_image` | Example PDF processing | Upload example with "init-" prefix |
| 1287 | `upload_pdf_image` | Image upload processing | Store uploaded image |
| 1382 | `upload_pdf_file` | PDF upload processing | Store original PDF |
| 1409 | `upload_pdf_image` | PDF to image conversion | Store PDF as image with "init-" prefix |
| 1474 | `upload_pdf_image` | Supplemental highlighting | Store highlighted version |
| 2191 | `upload_pdf_image` | Image upload (duplicate) | Store uploaded image |
| 2211 | `upload_pdf_image` | PDF to image (duplicate) | Store PDF as image |
| 2239 | `upload_pdf_image` | Bearings highlighting | Store highlighted version |

---

## GPT API Calls Analysis

### GPT Function Definitions

| Function | Line | Model | Purpose |
|----------|------|-------|---------|
| `extract_bearings_with_gpt()` | 191 | ft:gpt-3.5-turbo-0125:personal:ldr:BEoe3v67 | Extract bearings from legal text |
| `extract_supplemental_info_with_gpt()` | 993 | gpt-3.5-turbo-0125 | Extract property info (Land Lot, District, County) |

### Actual GPT API Calls

| Line | Function | Model | Context |
|------|----------|-------|---------|
| 273 | `client.chat.completions.create()` | Fine-tuned model | Inside `extract_bearings_with_gpt()` |
| 1006 | `client.chat.completions.create()` | GPT-3.5-turbo | Inside `extract_supplemental_info_with_gpt()` |

### GPT Function Call Hierarchy

| Line | Function Called | Parent Function | Called From | Purpose |
|------|----------------|----------------|-------------|---------|
| 1336 | `extract_supplemental_info_with_gpt()` | `process_image()` (Line 1270) | main → process_image() | Extract property info from image |
| 1350 | `extract_bearings_with_gpt()` | `process_image()` (Line 1270) | main → process_image() | Extract bearings from image |
| 1469 | `extract_supplemental_info_with_gpt()` | `process_pdf()` (Line 1373) | main → process_pdf() | Extract property info from PDF |
| 1498 | `extract_bearings_with_gpt()` | `process_pdf()` (Line 1373) | main → process_pdf() | Extract bearings from PDF |

---

## Refactoring Tasks

### Task 1: Consolidate GCS Upload Functions
**Priority: High**

- [ ] **Line 44**: Review `upload_pdf_preview()` - determine if UUID naming is still needed
- [ ] **Line 85**: Review `upload_pdf_file()` - can this be merged with `upload_pdf_image()`?
- [ ] **Line 125**: Review `upload_pdf_image()` - make this the primary upload function
- [ ] **Decision**: Keep one flexible upload function or maintain separate functions for different use cases?

### Task 2: Standardize GCS File Naming
**Priority: Medium**

- [ ] **Line 555**: Check if UUID naming pattern is necessary for highlighted previews
- [ ] **Line 805**: Review "init-" prefix usage for consistency
- [ ] **Line 1409**: Ensure consistent naming between PDF and image uploads
- [ ] **Line 1474**: Review "highlighted-" prefix usage
- [ ] **Line 2239**: Ensure consistent highlighting file naming

### Task 3: Eliminate Duplicate GCS Calls
**Priority: High**

- [ ] **Line 1287 vs 2191**: Same `upload_pdf_image` call for image processing - consolidate
- [ ] **Line 1409 vs 2215**: Same `upload_pdf_image` call for PDF to image - consolidate
- [ ] **Line 1474 vs 2239**: Same highlighting upload pattern - consolidate

### Task 4: Consolidate Process Functions
**Priority: High**

- [ ] **Line 1270**: Review `process_image()` function
- [ ] **Line 1373**: Review `process_pdf()` function
- [ ] **Analysis**: Both functions have nearly identical GPT call patterns
- [ ] **Decision**: Create shared processing function or keep separate?

### Task 5: Optimize GPT Call Patterns
**Priority: Medium**

- [ ] **Line 1336 + 1350**: `process_image()` makes 2 sequential GPT calls
- [ ] **Line 1469 + 1498**: `process_pdf()` makes 2 sequential GPT calls
- [ ] **Analysis**: Can these be combined into single GPT call with combined prompt?
- [ ] **Decision**: Keep separate calls or create combined extraction function?

### Task 6: Review Auto-Print Integration
**Priority: Low**

- [ ] **Line 1498**: Auto-print logic in `process_pdf()` - ensure it works with refactored GCS calls
- [ ] **Line 555**: Auto-print preview upload - ensure consistency with new patterns

---

## Recommended Refactoring Approach

### Phase 1: GCS Consolidation
1. Decide on single upload function vs specialized functions
2. Standardize file naming patterns across all uploads
3. Remove duplicate upload calls

### Phase 2: Process Function Consolidation
1. Extract common processing logic from `process_pdf()` and `process_image()`
2. Create shared helper functions for OCR → GPT → GCS flow
3. Maintain separate entry points but share implementation

### Phase 3: GPT Optimization
1. Analyze if supplemental info and bearings extraction can be combined
2. Consider single GPT call with structured output for both data types
3. Maintain backward compatibility with existing session state

### Phase 4: Testing & Validation
1. Ensure all existing functionality works after refactoring
2. Verify GCS bucket organization remains consistent
3. Test auto-print functionality with new patterns

---

## Files to Review

- `utils/gcs_storage.py` - All upload functions
- `main.py` - Lines 555, 805, 1270-1520, 2191-2250
- Test the refactored code with actual PDF uploads
- Verify GCS bucket file organization after changes