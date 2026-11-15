"""Print the node graph of Comfy-Org's official Z-Image workflow templates.

    python scripts/inspect_official_templates.py [templates_dir]

The templates live inside the comfyui_workflow_templates_json package, i.e.
<ComfyUI>/python_embeded/Lib/site-packages/comfyui_workflow_templates_json/templates

This is where the ControlNet wiring in this repository came from: the official
Z-Image templates are the only published description of how
ModelPatchLoader + ZImageFunControlnet are supposed to be connected.
"""
import json
import os
import sys


def find_templates_dir(arg=None):
    if arg:
        return arg
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    # walk up looking for a ComfyUI install
    root = here
    for _ in range(6):
        candidates.append(os.path.join(root, "python_embeded", "Lib", "site-packages",
                                      "comfyui_workflow_templates_json", "templates"))
        root = os.path.dirname(root)
    candidates.append(os.path.join("C:\\", "ComfyUI", "ComfyUI_windows_portable_nvidia_cu128 (1)",
                                   "ComfyUI_windows_portable", "python_embeded", "Lib", "site-packages",
                                   "comfyui_workflow_templates_json", "templates"))
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def dump(wf, tag):
    for sg in (wf.get("definitions", {}) or {}).get("subgraphs", []) or []:
        print("### subgraph:", sg.get("name"))
        for n in sg.get("nodes", []):
            wv = json.dumps(n.get("widgets_values"), ensure_ascii=False)
            ins = [i.get("name") for i in (n.get("inputs") or [])]
            print("  %-28s %-48s %s" % (n.get("type"), wv[:46], ins))
    print("### top level:", tag)
    for n in wf.get("nodes", []):
        wv = json.dumps(n.get("widgets_values"), ensure_ascii=False)
        print("  %-28s %-48s" % (n.get("type"), wv[:46]))


def main():
    d = find_templates_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    if not d:
        print("templates dir not found; pass it as the first argument")
        return 1
    print("templates dir:", d)
    for name in ("image_z_image_turbo.json",
                 "image_z_image_turbo_fun_union_controlnet.json",
                 "image_z_image_turbo_int8.json"):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            print("--- missing:", name)
            continue
        print("---", name)
        dump(json.load(open(p, encoding="utf-8")), name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
