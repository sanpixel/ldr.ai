# ACD - Anti Code Drift Protocol

## Purpose
Prevent code drift by ensuring new features follow existing patterns instead of inventing new approaches.

## Protocol: Before Writing Any New Feature

### 1. Search for Similar Code
- Use `grepSearch` to find existing implementations
- Look for functions that do similar things
- Check how the codebase already solves this problem

### 2. Read and Understand the Pattern
- Use `readFile` or `readMultipleFiles` to study the existing code
- Understand why it works that way
- Note any important details or edge cases

### 3. Explain Your Plan
- Describe how you'll follow the existing pattern
- Point out which existing code you're copying from
- Explain any necessary differences
- **Wait for approval before coding**
- **Wait for approval before coding**
- **Wait for approval before coding**
- **Wait for approval before coding**
- **Wait for approval before coding**

### 4. Implement Consistently
- Use the same approach as the existing code
- Keep the same structure and flow
- Don't "improve" or "optimize" unless explicitly asked
- Match naming conventions and style

### 5. Verify Consistency
- Compare your new code to the existing pattern
- Ensure it follows the same flow
- Check for any unintentional differences

## Example: process_image() Mistake

**What Went Wrong:**
- `process_pdf()` already existed and worked
- Created `process_image()` with different file handling approach
- Added unnecessary complexity (file pointer manipulation)
- Caused crashes and connection issues

**What Should Have Happened:**
1. Search: "How does process_pdf handle files?"
2. Read: Study the PDF processing flow
3. Plan: "I'll use the same approach - open file, OCR, extract bearings"
4. Implement: Copy the working pattern
5. Result: No crashes, consistent codebase

## Key Principle
**If it already works somewhere, use that approach. Don't reinvent.**
