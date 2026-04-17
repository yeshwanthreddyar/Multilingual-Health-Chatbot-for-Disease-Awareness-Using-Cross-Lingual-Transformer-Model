"""
Run this script from your healthbot folder:
    cd C:\Users\Yeshwanth Reddy A R\Downloads\healthbot
    python fix_syntax_error.py

It will fix the syntax error in app/dialog/manager.py automatically.
"""
import os
import re
import shutil

path = os.path.join("app", "dialog", "manager.py")

if not os.path.exists(path):
    print("ERROR: Could not find", path)
    print("Make sure you run this script from your healthbot folder.")
    input("Press Enter to exit.")
    raise SystemExit(1)

# Back up the broken file first
backup = path + ".bak"
shutil.copy2(path, backup)
print(f"Backed up original to {backup}")

with open(path, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

# Fix 1: any line that looks like:
#     factual_context = "
#     ".join(...)
# should be:
#     factual_context = "\n".join(...)
fixed = re.sub(
    r'factual_context\s*=\s*"[\r\n]+"\s*\.join\(([^)]+)\)',
    r'factual_context = "\\n".join(\1)',
    content,
)

# Fix 2: same pattern for any other broken "\n" that became a real newline inside a string
# Covers: "...\n" style in f-strings/prompts
fixed = re.sub(
    r'(?<=["\'])[\r\n]+(?=["\']\.)',
    r'\\n',
    fixed,
)

if fixed == content:
    print("Pattern not found — trying character-level scan...")
    lines = content.split("\n")
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect an unterminated string: odd number of unescaped quotes and no closing
        stripped = line.rstrip()
        if (
            'factual_context = "' in stripped
            and not stripped.endswith('"')
            and not stripped.endswith('",')
            and not stripped.endswith('")')
            and i + 1 < len(lines)
        ):
            # Merge this line with the next to close the string
            next_line = lines[i + 1].lstrip()
            merged = stripped + "\\n" + next_line
            new_lines.append(merged)
            i += 2
            print(f"  Fixed line {i}: merged broken string")
        else:
            new_lines.append(line)
            i += 1
    fixed = "\n".join(new_lines)

# Write back
with open(path, "w", encoding="utf-8") as f:
    f.write(fixed)
print(f"Written fixed file to {path}")

# Verify syntax
import ast
try:
    ast.parse(fixed)
    print("\n✅ SUCCESS — manager.py syntax is now valid.")
    print("You can now run: python main.py")
except SyntaxError as e:
    print(f"\n❌ Still has syntax error at line {e.lineno}: {e.msg}")
    print("Restoring backup...")
    shutil.copy2(backup, path)
    print(f"Restored from {backup}")
    print("\nPlease share the error message and we will fix it manually.")

input("\nPress Enter to exit.")
