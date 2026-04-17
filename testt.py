import re, ast

with open('app/dialog/manager.py', 'rb') as f:
    data = f.read()

# Fix ALL bare newlines inside string literals (the \n that became real newlines)
fixed = re.sub(rb'(['\''\"]) *\r?\n *(['\''\"]\s*[\+\.])', rb'\1\\n\2', data)

# Also fix the specific pattern: = \"\r\n\".join
fixed = re.sub(rb'= \"\r?\n\"\.join', rb'= \"\\n\".join', fixed)

# Fix multiline string in prompt: sentence.\r\n\" -> sentence.\\n\"
fixed = re.sub(rb'symptoms\.\r?\n\s*\"', rb'symptoms.\\n\"', fixed)
fixed = re.sub(rb'\\\\n\\\\n\"\r?\n\s*\+', rb'\\\\n\\\\n\" +', fixed)
fixed = re.sub(rb'\.\r?\n\s*f\"', rb'. f\"', fixed)

with open('app/dialog/manager.py', 'wb') as f:
    f.write(fixed)
print('Done')