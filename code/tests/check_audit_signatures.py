import json
import ast
import sys

nb_path = "d:\\do_an_2_clustering\\code\\check_code_gpt.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

code_parts = []
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        code_parts.append(src)

full_code = "\n\n".join(code_parts)

try:
    tree = ast.parse(full_code)
except SyntaxError as e:
    print("SYNTAX ERROR while parsing notebook code:")
    print(e)
    sys.exit(2)

funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

for fname in [
    "audit_nonnegative_constraints",
    "audit_soft_positive_columns",
    "run_phase_2_data_audit",
]:
    node = funcs.get(fname)
    if node is None:
        print(f"MISSING: {fname}")
    else:
        args = [a.arg for a in node.args.args]
        print(f"{fname} args: {args}")

print("Parsing OK")
