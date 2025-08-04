# Cursor AI Prompting Framework

This repository provides a structured set of prompting rules to optimize interactions with Cursor AI. It includes three key files to guide the AI’s behavior across various coding tasks.

## Files and Their Roles

### **`core.md`**
- **Purpose**: Establishes foundational rules for consistent AI behavior across all tasks.
- **Usage**: Place this file in your project’s `.cursor/rules/` folder to apply it persistently:
  - Save `core.md` under `.cursor/rules/` in the workspace root.
  - Cursor automatically applies rules from this folder to all AI interactions.
- **When to Use**: Always include as the base configuration for reliable, codebase-aware assistance.

### **`refresh.md`**
- **Purpose**: Directs the AI to diagnose and fix persistent issues, such as bugs or errors.
- **Usage**: Use as a situational prompt:
  - Copy the contents of `refresh.md`.
  - Replace `{my query}` with your specific issue (e.g., "the login button still crashes").
  - Paste into Cursor’s AI input (chat or composer).
- **When to Use**: Apply when debugging or resolving recurring problems—e.g., “It’s still broken after the last fix.”

### **`request.md`**
- **Purpose**: Guides the AI to implement new features or modify existing code.
- **Usage**: Use as a situational prompt:
  - Copy the contents of `request.md`.
  - Replace `{my request}` with your task (e.g., "add a save button").
  - Paste into Cursor’s AI input.
- **When to Use**: Apply for starting development tasks—e.g., “Build feature X” or “Update function Y.”

## Setup Instructions

1. **Clone or Download**: Get this repository locally.
2. **Configure Core Rules**:
   - Create a `.cursor/rules/` folder in your project’s root (if it doesn’t exist).
   - Copy `core.md` into `.cursor/rules/` to set persistent rules.
3. **Apply Situational Prompts**:
   - For debugging: Use `refresh.md` by copying, editing `{my query}`, and submitting.
   - For development: Use `request.md` by copying, editing `{my request}`, and submitting.

## Usage Tips

- **Project Rules**: The `.cursor/rules/` folder is Cursor’s modern system (replacing the legacy `.cursorrules` file). Add additional rule files here as needed.
- **Placeholders**: Always replace `{my query}` or `{my request}` with specific details before submitting prompts.
- **Adaptability**: These rules are optimized for Cursor AI but can be tweaked for other AI tools with similar capabilities.

## Notes

- Ensure file paths in prompts (e.g., for `edit_file`) are relative to the workspace root, per `core.md`.
- Test prompts in small steps to verify AI behavior aligns with your project’s needs.
- Contributions or suggestions to improve this framework are welcome!