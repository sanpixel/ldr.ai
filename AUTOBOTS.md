# AUTOBOTS - Automated Testing Steps

## Updated Testing Steps for Database Metrics Implementation:

1. **Implement 2 database metrics** 
   - Add new fields to reasoning_data in `extract_bearings_with_gpt` function
   - Ensure metrics are properly passed as parameters

2. **Commit and push - triggers auto-deploy**
   ```powershell
   git add .
   git commit -m "Add [metric_names] to reasoning data"
   git push
   ```

3. **Wait ~200 seconds for deployment**
   ```powershell
   gh run list
   ```
   - Wait until ✓ appears for the deployment

4. **Open site in browser**
   ```powershell
   Start-Process "https://ldr.clocknumbers.com/"
   ```

5. **Wait 44 seconds for auto-processing + database write**
   - Auto-processing will trigger on the first local PDF
   - Database write occurs automatically

6. **Query database to verify new metrics**
   ```powershell
   $env:SUPABASE_URL="https://xvlzjyjqqgfpcxqnplds.supabase.co"; $env:SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inh2bHpqeWpxcWdmcGN4cW5wbGRzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDY4MjYwMSwiZXhwIjoyMDcwMjU4NjAxfQ.XaMGGB8_Bb3UEPNwRXSkddHSijkWNDCfJf9NmV0xlHc"; python check_db.py
   ```

7. **Verify all fields are populated in the reasoning_data JSON**
   - Check that new metrics appear in the database entry
   - Ensure values are correct and not null

8. **If metrics are missing/null - FIX FIRST, then RETEST**
   - Don't proceed to next metrics if current ones failed
   - Debug the issue (check parameter passing, function calls, etc.)
   - Fix the code, commit, push, wait for deployment
   - Repeat steps 4-7 until metrics show up correctly
   - Only then proceed to implement next metrics

9. **Check generated columns are working (if needed)**
   - Verify database schema updates are working properly

## Metrics Implementation Progress:

**Completed (6/21):**
- ✅ file_size (#10: PDF size in bytes)
- ✅ debug_mode (#16: Was debug mode enabled?)
- ✅ processing_time (#1: How long GPT analysis took)
- ✅ page_count (#2: Number of PDF pages processed)
- ✅ text_length (#3: Characters extracted via OCR)
- ✅ bearing_count (#4: Number of bearings found)

**Next Implementation (2 at a time):**
- [ ] parsed_bearing_count (#5: Number successfully parsed to coordinates)
- [ ] parsing_success_rate (#6: % of bearings that parsed correctly)
- [ ] ocr_confidence (#7: OCR clarity score)
- [ ] model_version (#8: Which GPT model used)
- [ ] supplemental_info_found (#9: Boolean: found Land Lot/County data?)
- [ ] file_hash (#11: MD5/SHA256 for duplicate detection)
- [ ] original_filename (#12: User's original filename)
- [ ] storage_path (#13: Where file is stored)
- [ ] upload_method (#14: "file_upload", "local_file", "google_drive")
- [ ] session_id (#15: Browser session tracking)
- [ ] ip_address (#17: User's IP for analytics)
- [ ] user_agent (#18: Browser info)
- [ ] prompt_version (#19: Which prompt template was used)
- [ ] temperature (#20: GPT temperature setting - 0.1)
- [ ] max_tokens (#21: Token limits)

**Total: 21 metrics to implement**

## Notes:
- Follow steps in exact order
- Don't skip the wait times
- Each metric implementation should be done 2 at a time for manageable testing
- Continue until all 21 metrics are implemented
- Update progress after each successful deployment
