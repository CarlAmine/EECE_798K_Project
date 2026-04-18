import json
import glob
import os

paths = glob.glob(r'c:\Users\carla\Desktop\EECE 798K\Project\notebooks\*.ipynb')

for path in sorted(paths):
    print(f"\n{'='*50}\nNotebook: {os.path.basename(path)}\n{'='*50}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            cells = nb.get('cells', [])
            
            output_found = False
            for i, cell in enumerate(cells):
                if cell.get('cell_type') == 'code':
                    outputs = cell.get('outputs', [])
                    for out in outputs:
                        if out.get('output_type') == 'stream':
                            print(f"--- Cell {i} Text Output ---")
                            print("".join(out.get('text', [])[:15]) + ("...\n" if len(out.get('text', [])) > 15 else ""))
                            output_found = True
                        elif out.get('output_type') == 'execute_result' or out.get('output_type') == 'display_data':
                            data = out.get('data', {})
                            if 'text/plain' in data:
                                print(f"--- Cell {i} Data Output ---")
                                print("".join(data['text/plain'][:15]) + ("...\n" if len(data['text/plain']) > 15 else ""))
                                output_found = True
                            if 'image/png' in data:
                                print(f"--- Cell {i} Output ---")
                                print("[Image/Plot generated]")
                                output_found = True
            if not output_found:
                print("No text or image outputs found.")
    except Exception as e:
        print(f"Error parsing {path}: {e}")
