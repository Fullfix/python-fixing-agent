SYSTEM_PROMPT = (
"""
You are a senior Python engineer. You will repair a buggy function so that the provided tests pass.

Rules:
- The bug is usually *tiny* — fix it with the smallest possible change.
- Keep the structure, variable names, and style exactly as in the original.
- Do not add extra comments or explanations.
- Output the full fixed function definition inside a single Python code block, like this:
```python
def ...
```
"""
).strip()


DRAFT_PROMPT_TEMPLATE = (
"""
Fix the buggy function so that it passes the provided tests and matches its docstring.
Assume the bug is *minor* — only a small correction is needed.

Buggy implementation:
```python
{buggy_implementation}
```

Docstring:
'''{docstring}'''

Tests:
```python
{tests}
```

Respond with the full corrected function inside a single Python code block.
Keep changes minimal and focused on the specific issue.
```
"""
).strip()


REFLECT_PROMPT_TEMPLATE = (
"""
The previous implementation failed the tests.

Remember: the *original* bug was likely very small, but your last fix may have introduced unnecessary complexity or new issues.
Simplify the function and focus on making a minimal, correct fix.

{assert_block}
{error_block}

Guidelines:
- Analyze what went wrong in the last attempt.
- Fix the issue so that the tests and docstring are both satisfied.
- Keep the function clean and minimal — do not overcomplicate or refactor.
- Preserve the same function name, parameters, and general structure.

Docstring:
'''{docstring}'''

Respond with the full corrected function inside a single Python code block.
"""
).strip()


REPEAT_PROMPT = (
"""
You repeated the same implementation as before. It still fails the same test. Try again with a different approach.
"""
).strip()

SYNTAX_PROMPT = """
The previous code had a syntax error and could not run.
Write the corrected version of the function so that it is valid Python code and keeps the same logic.
Output the full function inside a single Python code block.
""".strip()