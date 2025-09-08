python - <<'PY'
import os, pathlib
root='.'
for path, dirs, files in os.walk(root):
    depth = pathlib.Path(path).relative_to(root).parts
    indent = '    ' * len(depth)
    if path != '.': print(f"{indent[:-4]}└── {path.split('/')[-1]}/")
    for f in sorted(files):
        if f == os.path.basename(__file__): continue
        print(f"{indent}    {f}")
PY
