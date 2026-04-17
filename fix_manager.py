import re, shutil
p = "app/dialog/manager.py"
shutil.copy2(p, p + ".bak")
with open(p, "rb") as f:
    data = f.read()
fixed = re.sub(b"= \"\\r?\\n\"\\.join", b"= \"\\\\n\".join", data)
fixed = re.sub(b"symptoms\\.\\r?\\n\\s*\"", b"symptoms.\\\\n\"", fixed)
fixed = re.sub(b"([a-z]\\.)(\\r?\\n)(\\s*f\")", b"\\1 \\3", fixed)
fixed = re.sub(b"(\\\\n\\\\n\")(\\r?\\n)(\\s*\\+)", b"\\1 \\3", fixed)
with open(p, "wb") as f:
    f.write(fixed)
import ast
try:
    ast.parse(fixed.decode("utf-8", errors="replace"))
    print("SUCCESS - manager.py is fixed. Now run: python main.py")
except SyntaxError as e:
    print("Line " + str(e.lineno) + " still broken: " + str(e.msg))
    print("Restoring backup...")
    shutil.copy2(p + ".bak", p)
