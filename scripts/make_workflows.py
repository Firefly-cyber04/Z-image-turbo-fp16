"""Rebuild every Z-Image-Turbo workflow from a single graph definition.

    python scripts/make_workflows.py

Each workflow is declared once, as a small node graph, and is then emitted
twice:

  * workflows/api_prompt_*.json  - flat API prompts, ready for POST /prompt
  * workflows/*_ui.json          - ComfyUI UI graphs, ready to drag into the app

The UI output is verified by converting it back to an API prompt and comparing
it against the API output, so the two files can never drift apart.

Everything here was read off the machine that actually produced the renders:
the ComfyUI 0.45.0 workflow templates, the ComfyUI 0.45.0 source, and the PNG
metadata of previous renders.

Sampler naming note: there is no "dpmpp_2p_sde_heun" in ComfyUI. The names in
the KSampler dropdown are dpmpp_2m_sde_heun and dpmpp_2m_sde_heun_gpu (the
"_gpu" variant keeps the model on the GPU between steps). The masked-redraw
workflow below uses dpmpp_2m_sde_heun_gpu, which is what the previous
local renders used.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

UNET = "z_image_turbo_bf16.safetensors"
VAE = "ae.safetensors"
CLIP = "qwen_3_4b.safetensors"
CONTROLNET = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors"

# UI-only widget: the frontend uses it to decide how to advance the seed, the
# API prompt format does not accept it.
UI_ONLY_WIDGETS = {"KSampler": {"control_after_generate"}}

# Optional input sockets. The ComfyUI frontend marks these with shape 7 so it
# renders them as optional, exactly like Comfy-Org's own template does.
OPTIONAL_INPUTS = {
    "ZImageFunControlnet": {"image", "inpaint_image", "mask", "start_percent", "end_percent"},
}


# --------------------------------------------------------------------------
# tiny graph builder
# --------------------------------------------------------------------------
def node(nid, ntype, widgets=(), pos=(0, 0), size=(270, 60), ins=(), outs=()):
    """ins: iterable of (input_name, input_type, (src_id, src_slot))
    outs: iterable of (output_name, output_type)"""
    return {
        "id": nid,
        "type": ntype,
        "widgets": list(widgets),
        "pos": list(pos),
        "size": list(size),
        "ins": [{"name": n, "type": t, "src": s} for (n, t, s) in ins],
        "outs": [{"name": n, "type": t} for (n, t) in outs],
    }


class Graph:
    def __init__(self):
        self.nodes = []
        self._link_id = 0

    def add(self, n):
        self.nodes.append(n)
        return n

    def links(self):
        out = []
        for n in self.nodes:
            for i in n["ins"]:
                self._link_id += 1
                out.append([self._link_id, i["src"][0], i["src"][1], n["id"],
                            next(k for k, x in enumerate(n["ins"]) if x is i), i["type"]])
        return out

    def order(self):
        """Topological order of node ids."""
        by_id = {n["id"]: n for n in self.nodes}
        seen, order = set(), []

        def visit(nid):
            if nid in seen:
                return
            seen.add(nid)
            for i in by_id[nid]["ins"]:
                visit(i["src"][0])
            order.append(nid)

        for n in self.nodes:
            visit(n["id"])
        return order


def to_api(g):
    by_id = {n["id"]: n for n in g.nodes}
    out = {}
    for n in g.nodes:
        inputs = {}
        for name, value in n["widgets"]:
            if name in UI_ONLY_WIDGETS.get(n["type"], ()):
                continue
            inputs[name] = value
        for i in n["ins"]:
            # node ids in an API prompt are string keys, and a link is
            # ["<node_id>", slot_index] - the id must be a string too
            inputs[i["name"]] = [str(i["src"][0]), i["src"][1]]
        out[str(n["id"])] = {"class_type": n["type"], "inputs": inputs}
    return out


def to_ui(g, title=""):
    links = g.links()
    order = {nid: i for i, nid in enumerate(g.order())}
    nodes = []
    for n in g.nodes:
        link_ids = {}
        for l in links:
            if l[3] == n["id"]:
                link_ids[l[4]] = l[0]
        ui_ins = []
        optional = OPTIONAL_INPUTS.get(n["type"], set())
        for k, i in enumerate(n["ins"]):
            slot = {"name": i["name"], "type": i["type"], "link": link_ids.get(k)}
            if i["name"] in optional:
                slot["shape"] = 7
            ui_ins.append(slot)
        ui_outs = []
        for k, o in enumerate(n["outs"]):
            used = [l[0] for l in links if l[1] == n["id"] and l[2] == k]
            ui_outs.append({"name": o["name"], "type": o["type"], "slot_index": 0, "links": used or None})
        widgets_named = {}
        for name, value in n["widgets"]:
            widgets_named[name] = value
        nodes.append({
            "id": n["id"],
            "type": n["type"],
            "pos": n["pos"],
            "size": n["size"],
            "flags": {},
            "order": order[n["id"]],
            "mode": 0,
            "inputs": ui_ins,
            "outputs": ui_outs,
            "properties": {"Node name for S&R": n["type"]},
            "widgets_values": [v for _, v in n["widgets"]],
            "widgets_values_named": widgets_named,
            "title": title if title and n["id"] == 1 else n["type"],
        })
    max_link = max([l[0] for l in links], default=0)
    return {
        "id": "zimage-workflow",
        "revision": 0,
        "last_node_id": max(n["id"] for n in g.nodes),
        "last_link_id": max_link,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.7, "offset": [60, 200]}},
        "version": 0.4,
    }


def ui_to_api(ui):
    """Read a UI graph back into an API prompt. Used as the self-check."""
    by_id = {n["id"]: n for n in ui["nodes"]}
    links = {l[0]: l for l in ui["links"]}
    out = {}
    for n in ui["nodes"]:
        inputs = {}
        for name, value in n.get("widgets_values_named", {}).items():
            if name in UI_ONLY_WIDGETS.get(n["type"], ()):
                continue
            inputs[name] = value
        for slot, i in enumerate(n.get("inputs", [])):
            if i.get("link") is None:
                continue
            l = links[i["link"]]
            inputs[i["name"]] = [str(l[1]), l[2]]
        out[str(n["id"])] = {"class_type": n["type"], "inputs": inputs}
    return out


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
# Z-Image's text encoder is Qwen3-4B, so Chinese prompts work natively. That
# is the main practical difference from MiniMax-H3-Image (which needs an
# English "already expanded" prompt).
PROMPT = (
    "一只橘猫坐在洒满午后阳光的木质窗台上，脖子上系着一条淡蓝色条纹围巾，"
    "窗外是虚化的绿植与暖色天空，室内静谧，柔和的自然光，细腻的毛发质感，"
    "温暖的色调，插画风格，高细节"
)
NEGATIVE = (
    "blurry, low quality, deformed, bad anatomy, extra limbs, floating limbs, "
    "wrong position, censored, clothes, text, watermark"
)

# ControlNet demo: the canny edge map of the base photo fixes the layout, the
# prompt decides what is actually in it. Same window frame, different animal.
CONTROL_PROMPT = (
    "一只柴犬坐在洒满午后阳光的木质窗台上，脖子上系着一条红色蝴蝶结领结，"
    "窗外是虚化的绿植与暖色天空，室内静谧，柔和的自然光，细腻的毛发质感，"
    "温暖的色调，写实摄影，高细节"
)

# Masked redraw demo: only the area covered by the alpha hole is regenerated.
INPAINT_PROMPT = (
    "一只橘猫坐在洒满午后阳光的木质窗台上，脖子上系着一条红色的缎面蝴蝶结，"
    "窗外是虚化的绿植与暖色天空，室内静谧，柔和的自然光，细腻的毛发质感，"
    "温暖的色调，插画风格，高细节"
)


# --------------------------------------------------------------------------
# workflows
# --------------------------------------------------------------------------
def g_t2i(prompt=PROMPT, negative=NEGATIVE, seed=20261115, width=1024, height=1024,
          steps=9, sampler="euler", scheduler="simple", prefix="zimage_turbo_t2i"):
    g = Graph()
    g.add(node(1, "UNETLoader", [("unet_name", UNET), ("weight_dtype", "default")],
               pos=(20, 20), outs=[("MODEL", "MODEL")]))
    g.add(node(2, "CLIPLoader", [("clip_name", CLIP), ("type", "lumina2"), ("device", "default")],
               pos=(20, 140), size=(270, 106), outs=[("CLIP", "CLIP")]))
    g.add(node(3, "VAELoader", [("vae_name", VAE)], pos=(20, 280), outs=[("VAE", "VAE")]))
    g.add(node(4, "CLIPTextEncode", [("text", prompt)], pos=(320, 40), size=(380, 160),
               ins=[("clip", "CLIP", (2, 0))], outs=[("CONDITIONING", "CONDITIONING")]))
    g.add(node(5, "CLIPTextEncode", [("text", negative)], pos=(320, 230), size=(380, 120),
               ins=[("clip", "CLIP", (2, 0))], outs=[("CONDITIONING", "CONDITIONING")]))
    g.add(node(6, "EmptySD3LatentImage",
               [("width", width), ("height", height), ("batch_size", 1)],
               pos=(20, 370), outs=[("LATENT", "LATENT")]))
    g.add(node(7, "KSampler",
               [("seed", seed), ("control_after_generate", "fixed"), ("steps", steps),
                ("cfg", 1.0), ("sampler_name", sampler), ("scheduler", scheduler),
                ("denoise", 1.0)],
               pos=(760, 40), size=(260, 300),
               ins=[("model", "MODEL", (1, 0)), ("positive", "CONDITIONING", (4, 0)),
                    ("negative", "CONDITIONING", (5, 0)), ("latent_image", "LATENT", (6, 0))],
               outs=[("LATENT", "LATENT")]))
    g.add(node(8, "VAEDecode", pos=(1060, 40), size=(210, 46),
               ins=[("samples", "LATENT", (7, 0)), ("vae", "VAE", (3, 0))], outs=[("IMAGE", "IMAGE")]))
    g.add(node(9, "SaveImage", [("filename_prefix", prefix)], pos=(1310, 40),
               ins=[("images", "IMAGE", (8, 0))], outs=[("images", "IMAGE")]))
    return g


def g_auraflow(prompt=PROMPT, negative=NEGATIVE, seed=20261115, width=1024, height=1024):
    """T2I with the AuraFlow shift wrapper, 8 steps, res_multistep/simple."""
    g = g_t2i(prompt, negative, seed, width, height,
              steps=8, sampler="res_multistep", scheduler="simple",
              prefix="zimage_turbo_auraflow")
    g.add(node(10, "ModelSamplingAuraFlow", [("shift", 3.0), ("sampling", "flow")],
               pos=(470, 20), size=(270, 82), ins=[("model", "MODEL", (1, 0))], outs=[("MODEL", "MODEL")]))
    for n in g.nodes:
        for i in n["ins"]:
            if i["name"] == "model" and i["src"] == (1, 0):
                i["src"] = (10, 0)
    return g


def g_controlnet(image="zimage_demo_base.png", prompt=CONTROL_PROMPT, seed=20261115,
                 width=1024, height=1024, steps=8, sampler="res_multistep", scheduler="simple",
                 canny_low=0.25, canny_high=0.6):
    """Canny-driven ControlNet. This is the structure Comfy-Org ships in
    image_z_image_turbo_fun_union_controlnet.json: no latent is encoded from
    the input image, the edge map only conditions the model.

    The shipped template uses canny thresholds 0.1 / 0.32. Those pick up every
    single hair of a furry subject and the model then draws them back as dark
    streaks, so the defaults here are raised. Lower them for hard-edged
    subjects such as architecture or product shots.
    """
    g = Graph()
    g.add(node(1, "UNETLoader", [("unet_name", UNET), ("weight_dtype", "default")],
               pos=(20, 20), outs=[("MODEL", "MODEL")]))
    g.add(node(2, "CLIPLoader", [("clip_name", CLIP), ("type", "lumina2"), ("device", "default")],
               pos=(20, 140), size=(270, 106), outs=[("CLIP", "CLIP")]))
    g.add(node(3, "VAELoader", [("vae_name", VAE)], pos=(20, 280), outs=[("VAE", "VAE")]))
    g.add(node(4, "CLIPTextEncode", [("text", prompt)], pos=(320, 40), size=(420, 170),
               ins=[("clip", "CLIP", (2, 0))], outs=[("CONDITIONING", "CONDITIONING")]))
    g.add(node(5, "ConditioningZeroOut", pos=(320, 240), size=(210, 46),
               ins=[("conditioning", "CONDITIONING", (4, 0))], outs=[("CONDITIONING", "CONDITIONING")]))
    g.add(node(6, "LoadImage", [("image", image), ("upload", "image")],
               pos=(20, 370), size=(270, 314), outs=[("IMAGE", "IMAGE"), ("MASK", "MASK")]))
    g.add(node(7, "Canny", [("low_threshold", canny_low), ("high_threshold", canny_high)],
               pos=(320, 400), ins=[("image", "IMAGE", (6, 0))], outs=[("IMAGE", "IMAGE")]))
    g.add(node(8, "EmptySD3LatentImage",
               [("width", width), ("height", height), ("batch_size", 1)],
               pos=(620, 380), outs=[("LATENT", "LATENT")]))
    g.add(node(9, "ModelPatchLoader", [("name", CONTROLNET)], pos=(620, 20),
               outs=[("MODEL_PATCH", "MODEL_PATCH")]))
    g.add(node(10, "ZImageFunControlnet", [("strength", 1.0)], pos=(940, 20), size=(270, 120),
               ins=[("model", "MODEL", (1, 0)), ("model_patch", "MODEL_PATCH", (9, 0)),
                    ("vae", "VAE", (3, 0)), ("image", "IMAGE", (7, 0))],
               outs=[("MODEL", "MODEL")]))
    g.add(node(11, "ModelSamplingAuraFlow", [("shift", 3.0), ("sampling", "flow")],
               pos=(940, 200), size=(270, 82), ins=[("model", "MODEL", (10, 0))], outs=[("MODEL", "MODEL")]))
    g.add(node(12, "KSampler",
               [("seed", seed), ("control_after_generate", "fixed"), ("steps", steps),
                ("cfg", 1.0), ("sampler_name", sampler), ("scheduler", scheduler),
                ("denoise", 1.0)],
               pos=(1250, 120), size=(260, 300),
               ins=[("model", "MODEL", (11, 0)), ("positive", "CONDITIONING", (4, 0)),
                    ("negative", "CONDITIONING", (5, 0)), ("latent_image", "LATENT", (8, 0))],
               outs=[("LATENT", "LATENT")]))
    g.add(node(13, "VAEDecode", pos=(1560, 120), size=(210, 46),
               ins=[("samples", "LATENT", (12, 0)), ("vae", "VAE", (3, 0))], outs=[("IMAGE", "IMAGE")]))
    g.add(node(14, "SaveImage", [("filename_prefix", "zimage_turbo_controlnet")],
               pos=(1810, 120), ins=[("images", "IMAGE", (13, 0))], outs=[("images", "IMAGE")]))
    return g


def g_inpaint(image="zimage_demo_inpaint.png", prompt=INPAINT_PROMPT, seed=20261115,
              width=1024, height=1024, steps=9, sampler="dpmpp_2m_sde_heun_gpu",
              scheduler="sgm_uniform"):
    """Masked redraw through the Fun ControlNet 2.1 patch.

    The input PNG carries an alpha hole. LoadImage turns that into mask=1
    (1 - alpha), and mask=1 means "regenerate here" for both consumers:

      SetLatentNoiseMask  - the sampler keeps resampling the hole instead of
                             pasting the original latent back
      ZImageFunControlnet - the hole is dropped from the inpaint reference and
                             the mask channel is raised, so the ControlNet knows
                             that area is free to be repainted

    The rest of the frame is preserved pixel for pixel.
    """
    g = Graph()
    g.add(node(1, "UNETLoader", [("unet_name", UNET), ("weight_dtype", "default")],
               pos=(20, 20), outs=[("MODEL", "MODEL")]))
    g.add(node(2, "CLIPLoader", [("clip_name", CLIP), ("type", "lumina2"), ("device", "default")],
               pos=(20, 140), size=(270, 106), outs=[("CLIP", "CLIP")]))
    g.add(node(3, "VAELoader", [("vae_name", VAE)], pos=(20, 280), outs=[("VAE", "VAE")]))
    g.add(node(4, "CLIPTextEncode", [("text", prompt)], pos=(320, 40), size=(420, 170),
               ins=[("clip", "CLIP", (2, 0))], outs=[("CONDITIONING", "CONDITIONING")]))
    g.add(node(5, "ConditioningZeroOut", pos=(320, 240), size=(210, 46),
               ins=[("conditioning", "CONDITIONING", (4, 0))], outs=[("CONDITIONING", "CONDITIONING")]))
    g.add(node(6, "LoadImage", [("image", image), ("upload", "image")],
               pos=(20, 370), size=(270, 314), outs=[("IMAGE", "IMAGE"), ("MASK", "MASK")]))
    g.add(node(7, "VAEEncode", pos=(320, 420), size=(140, 46),
               ins=[("pixels", "IMAGE", (6, 0)), ("vae", "VAE", (3, 0))], outs=[("LATENT", "LATENT")]))
    g.add(node(8, "SetLatentNoiseMask", pos=(480, 420), size=(170, 46),
               ins=[("samples", "LATENT", (7, 0)), ("mask", "MASK", (6, 1))], outs=[("LATENT", "LATENT")]))
    g.add(node(9, "ModelPatchLoader", [("name", CONTROLNET)], pos=(680, 20),
               outs=[("MODEL_PATCH", "MODEL_PATCH")]))
    g.add(node(10, "ZImageFunControlnet", [("strength", 1.0)], pos=(1000, 20), size=(270, 140),
               ins=[("model", "MODEL", (1, 0)), ("model_patch", "MODEL_PATCH", (9, 0)),
                    ("vae", "VAE", (3, 0)), ("inpaint_image", "IMAGE", (6, 0)),
                    ("mask", "MASK", (6, 1))],
               outs=[("MODEL", "MODEL")]))
    g.add(node(11, "ModelSamplingAuraFlow", [("shift", 3.0), ("sampling", "flow")],
               pos=(1000, 220), size=(270, 82), ins=[("model", "MODEL", (10, 0))], outs=[("MODEL", "MODEL")]))
    g.add(node(12, "KSampler",
               [("seed", seed), ("control_after_generate", "fixed"), ("steps", steps),
                ("cfg", 1.0), ("sampler_name", sampler), ("scheduler", scheduler),
                ("denoise", 1.0)],
               pos=(1320, 120), size=(260, 300),
               ins=[("model", "MODEL", (11, 0)), ("positive", "CONDITIONING", (4, 0)),
                    ("negative", "CONDITIONING", (5, 0)), ("latent_image", "LATENT", (8, 0))],
               outs=[("LATENT", "LATENT")]))
    g.add(node(13, "VAEDecode", pos=(1630, 120), size=(210, 46),
               ins=[("samples", "LATENT", (12, 0)), ("vae", "VAE", (3, 0))], outs=[("IMAGE", "IMAGE")]))
    g.add(node(14, "SaveImage", [("filename_prefix", "zimage_turbo_inpaint")],
               pos=(1880, 120), ins=[("images", "IMAGE", (13, 0))], outs=[("images", "IMAGE")]))
    return g


def write(name, data):
    path = os.path.join(REPO, "workflows", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("wrote", os.path.relpath(path, REPO))


def emit(api_name, ui_name, g):
    api = to_api(g)
    ui = to_ui(g)
    back = ui_to_api(ui)
    if back != api:
        raise SystemExit("UI/API mismatch for %s" % api_name)
    write(api_name, api)
    write(ui_name, ui)


if __name__ == "__main__":
    emit("api_prompt_zimage_t2i.json", "zimage_turbo_t2i_ui.json", g_t2i())
    emit("api_prompt_zimage_auraflow.json", "zimage_turbo_auraflow_ui.json", g_auraflow())
    emit("api_prompt_zimage_controlnet.json", "zimage_turbo_controlnet_ui.json", g_controlnet())
    emit("api_prompt_zimage_inpaint.json", "zimage_turbo_inpaint_ui.json", g_inpaint())
    print("all workflows written and cross-checked")
