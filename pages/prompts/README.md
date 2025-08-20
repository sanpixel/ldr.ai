# Prompt Version History

This directory contains all versions of the bearings prompt for tracking changes and enabling rollbacks.

## Current Version: v1.0

**File:** `bearings_prompt_v1.0.txt`  
**Date:** 2025-08-20  
**Status:** Active  

## Versioning Strategy

- **Major versions** (X.0): Significant structural changes to classification categories or output format
- **Minor versions** (X.Y): Refinements to existing categories, new examples, improved instructions

## Version History

### v1.0 (2025-08-20)
**Initial structured classification system**

**Features:**
- Three-tier classification: `explicit_bearings`, `abstract_bearings`, `external_ref`
- Structured reasoning output: `CLASSIFICATION`, `CONFIDENCE`, `REASONING`, `EVIDENCE`
- `RANK_ALTERNATIVES` with ranked second/third choices
- Classification-based extraction logic handled by GPT
- Examples for each classification type

**Use Case:** Replaces hardcoded Python conditional logic with GPT-driven classification

## How to Update Prompts

1. **Backup current version:**
   ```bash
   copy bearings_prompt.txt pages\prompts\bearings_prompt_v[X.Y].txt
   ```

2. **Update main prompt:**
   ```bash
   # Edit bearings_prompt.txt with changes
   ```

3. **Document changes:**
   - Update this README.md with new version info
   - Note what changed and why
   - Update version number in main app if needed

4. **Test and commit:**
   ```bash
   git add .
   git commit -m "Update prompt to v[X.Y]: [description]"
   ```

## Rollback Instructions

To rollback to a previous version:

1. **Copy old version back:**
   ```bash
   copy pages\prompts\bearings_prompt_v[X.Y].txt bearings_prompt.txt
   ```

2. **Commit rollback:**
   ```bash
   git add bearings_prompt.txt
   git commit -m "Rollback prompt to v[X.Y] due to [reason]"
   ```

## Testing New Prompts

Use the reasoning dashboard at `/reasoning` to analyze prompt performance:
- Check classification accuracy
- Review confidence levels  
- Examine reasoning quality
- Look for edge cases and misclassifications

## Future Enhancements

Planned improvements:
- [ ] Mixed content handling (explicit + abstract)
- [ ] Curved/arc descriptions  
- [ ] International coordinate systems
- [ ] Multi-language support
- [ ] Confidence threshold tuning
