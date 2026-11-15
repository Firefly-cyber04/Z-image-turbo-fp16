"""Build the demo input assets for the Z-Image-Turbo workflows.

    python scripts/make_demo_assets.py --base samples/demo_base.png --out "<ComfyUI>/input"

Writes two files into --out:

    zimage_demo_base.png      plain RGB copy of the base photo
    zimage_demo_inpaint.png   the same photo with an alpha hole punched over the
                              region that should be repainted

Why an alpha hole and not a separate mask file: ComfyUI's LoadImage returns
``mask = 1 - alpha`` (see comfy/nodes.py, pillow_to_image). So alpha=0 becomes
mask=1, and mask=1 is what every downstream node means by "regenerate this
area":

  * SetLatentNoiseMask   -> denoise_mask=1 -> the sampler keeps regenerating
                            that area instead of pasting the original latent back
  * ZImageFunControlnet  -> mask channel = 1 -> the area behind the mask is
                            dropped from the inpaint reference (set to 0.5 grey)

Both conventions are the same, so a single PNG with an alpha hole drives the
whole masked-redraw workflow. A JPEG has no alpha channel, so its mask comes
back as all zeros and the sampler simply reproduces the input - that is the
classic "my ControlNet output is a copy of the source image" trap.

The polygon below is stored in pixel coordinates of the 1024x1024 base image.
"""
import argparse
import os
import shutil

from PIL import Image, ImageDraw

# Region of the demo base photo to repaint: the blue neckerchief.
# Coordinates are for the 1024x1024 base image.
HOLE_POLY = [
    (360, 545), (362, 448), (400, 390), (486, 344), (572, 340),
    (645, 350), (672, 392), (658, 438), (592, 462), (556, 528),
    (486, 572), (410, 572),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="source photo (1024x1024 RGB)")
    ap.add_argument("--out", required=True, help="ComfyUI input directory")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    base = Image.open(args.base).convert("RGB")
    if base.size != (1024, 1024):
        raise SystemExit("base image must be 1024x1024, got %s" % (base.size,))

    plain = os.path.join(out, "zimage_demo_base.png")
    base.save(plain, "PNG")
    print("wrote", plain)

    # RGBA copy with the hole punched out (alpha = 0 inside the polygon).
    rgba = base.convert("RGBA")
    alpha = Image.new("L", base.size, 255)
    ImageDraw.Draw(alpha).polygon(HOLE_POLY, fill=0)
    rgba.putalpha(alpha)

    hole = os.path.join(out, "zimage_demo_inpaint.png")
    rgba.save(hole, "PNG")
    print("wrote", hole)

    # A small preview so the hole placement can be eyeballed without running
    # the model: the hole is drawn in translucent magenta over the base photo.
    preview = base.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).polygon(HOLE_POLY, fill=(255, 0, 170, 110))
    preview = Image.alpha_composite(preview, overlay).convert("RGB")
    prev_path = os.path.join(out, "zimage_demo_inpaint_preview.png")
    preview.save(prev_path, "PNG")
    print("wrote", prev_path)


if __name__ == "__main__":
    main()
