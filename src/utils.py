import ast
import re


def extract_failed_assert(stderr: str) -> str | None:
    match = re.search(r'assert[^\n]+', stderr)
    if match:
        return match.group(0).strip()
    return None


def extract_error_summary(stderr: str, max_lines: int = 6) -> str:
    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def wrap_code_in_block(code: str | None) -> str:
    return f"```python\n{code or '<INVALID CODE GENERATED>'}\n```"


def extract_function_definition(text: str, name: str) -> str | None:
    m = re.search(r"```python\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    source_code = m.group(1) if m else text

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = source_code.splitlines()
            start_line = node.lineno - 1
            end_line = node.end_lineno
            func_code = "\n".join(lines[start_line:end_line])
            return func_code
    return None


def merge_declaration_with_definition(declaration: str, definition: str) -> str:
    imports_code = declaration.strip()
    function_code = definition.strip()

    match = re.search(r'^\s*def\s', imports_code, flags=re.MULTILINE)
    if match:
        prefix = imports_code[:match.start()].rstrip()
    else:
        prefix = imports_code

    combined = f"{prefix}\n\n\n{function_code}".strip() + "\n"
    return combined


def create_test_script(declaration: str, definition: str, test_script: str) -> str:
    return merge_declaration_with_definition(
        declaration=declaration,
        definition=definition
    ) + "\n" + test_script + "\n" + "print('ALL_PASS')"


def validate_test_script(exit_code: int, stdout: str) -> bool:
    return exit_code == 0 and "ALL_PASS" in stdout


def get_sampling_params(iteration_number: int) -> dict:
    base_temp = 0.25
    base_top_p = 0.9

    temperature = min(base_temp + 0.2 * iteration_number, 0.9)
    top_p = min(base_top_p + 0.03 * iteration_number, 0.98)

    return {
        "do_sample": True,
        "temperature": round(temperature, 3),
        "top_p": round(top_p, 3),
    }
