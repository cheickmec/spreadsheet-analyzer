#!/usr/bin/env python
"""Fix f-string escaping issues in tasks.py"""

import re


def fix_fstring_escaping():
    """Fix all f-string escaping issues in tasks.py"""

    with open("src/spreadsheet_analyzer/plugins/spreadsheet/tasks.py") as f:
        content = f.read()

    # Find all the code generation methods
    methods = ["_generate_import_code", "_generate_load_code", "_generate_profiling_code", "_generate_analysis_code"]

    # For each method, find the triple-quoted f-string content
    for method in methods:
        # Find method start
        method_pattern = rf'def {method}\(.*?\).*?:\s*""".*?"""\s*return f"""'
        match = re.search(method_pattern, content, re.DOTALL)
        if not match:
            continue

        # Find the start and end of the f-string
        start_pos = match.end()

        # Find the closing triple quotes
        end_match = re.search(r'"""', content[start_pos:])
        if not end_match:
            continue

        end_pos = start_pos + end_match.start()

        # Extract the f-string content
        fstring_content = content[start_pos:end_pos]

        # Fix all f-strings inside the main f-string
        # Replace {var} with {{var}} but not if already escaped
        fixed_content = fstring_content

        # Find all f-strings inside
        inner_fstrings = re.findall(r'(print\(f"[^"]+"\)|\.append\(f"[^"]+"\)|f"[^"]+")', fstring_content)

        for inner in inner_fstrings:
            # Extract the f-string content
            if 'print(f"' in inner:
                inner_content = inner[8:-2]  # Remove print(f" and ")
                prefix = 'print(f"'
                suffix = '")'
            elif '.append(f"' in inner:
                inner_content = inner[10:-2]  # Remove .append(f" and ")
                prefix = '.append(f"'
                suffix = '")'
            else:
                continue

            # Find all {expr} that are not already escaped {{expr}}
            def replace_braces(match):
                expr = match.group(1)
                # Skip if it's already escaped or if it's a method parameter
                if match.group(0).startswith("{{"):
                    return match.group(0)
                if expr in ["file_path", "sheet_name", "file_ext", "e"]:
                    # These should NOT be escaped as they're parameters
                    return match.group(0)
                return "{{" + expr + "}}"

            fixed_inner = re.sub(r"\{([^}]+)\}", replace_braces, inner_content)

            # Replace in the original content
            fixed_content = fixed_content.replace(inner, prefix + fixed_inner + suffix)

        # Also fix standalone dict/list constructions inside f-strings
        # Fix patterns like error_cells.append({ ... })
        fixed_content = re.sub(r"\.append\(\{([^}]+)\}\)", r".append({{\1}})", fixed_content)

        # Replace the content
        content = content[:start_pos] + fixed_content + content[end_pos:]

    # Write back
    with open("src/spreadsheet_analyzer/plugins/spreadsheet/tasks.py", "w") as f:
        f.write(content)

    print("Fixed f-string escaping issues")


if __name__ == "__main__":
    fix_fstring_escaping()
