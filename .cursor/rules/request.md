{my request (e.g., "add a save button")}

---

Design and implement the request described above using a systematic, validation-driven approach:

1. **Map System Context**:
   - Explore the codebase structure with `tree -L 4 --gitignore | cat` to locate where the feature belongs.
   - Identify relevant patterns, conventions, or domain models using `codebase_search` to ensure seamless integration.
   - Pinpoint integration points—e.g., UI components, data layers, or APIs—affected by the request.

2. **Specify Requirements**:
   - Break the request into clear, testable criteria—e.g., “Button triggers save, shows success state.”
   - Define use cases (normal and edge) and constraints (e.g., performance, UI consistency).
   - Set scope boundaries to keep the implementation focused and maintainable.

3. **Leverage Reusability**:
   - Search for existing components or utilities with `codebase_search` that can be adapted—e.g., a “button” component or “save” function.
   - Use `grep_search` to confirm similar implementations, ensuring consistency with project standards.
   - Evaluate if the feature could be abstracted for future reuse, noting potential opportunities.

4. **Plan Targeted Changes**:
   - List all files requiring edits (relative to workspace root), dependencies to update, and new files if needed.
   - Assess impacts on cross-cutting concerns—e.g., error handling, logging, or state management.
   - Balance immediate needs with long-term code health, planning minimal yet effective changes.

5. **Implement with Precision**:
   - Provide a step-by-step plan with specific code changes—include file paths, line numbers, and snippets.
   - Adhere to project conventions (e.g., naming, structure) and reuse existing patterns where applicable.
   - Highlight enhancements to organization or clarity—e.g., “Extract logic to a helper function.”

6. **Validate and Stabilize**:
   - Define test scenarios—e.g., “Save with valid data,” “Save with no input”—to confirm functionality.
   - Suggest validation methods: unit tests, UI checks, or logs, tailored to the project’s practices.
   - Recommend a stability check—e.g., “Monitor save API calls”—with rollback steps if issues arise.

This process delivers a well-integrated, reliable solution that enhances the codebase while meeting the request’s goals.