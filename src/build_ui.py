"""Build the single-file dashboard:  python -m src.build_ui   ->  app/index.html
Reads results/ui_payload.json (written by src.run_all) and inlines the fonts, so the file works offline."""
import base64
import json
import os

from .data import ROOT

APP = os.path.join(ROOT, "app")
FONTS = [("Caprasimo", 400, "caprasimo-latin"), ("Caprasimo", 400, "caprasimo-latin-ext"),
         ("Figtree", "400 700", "figtree-latin"), ("Figtree", "400 700", "figtree-latin-ext")]
RANGE = {"latin": "U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD",
         "ext": "U+0100-02BA,U+02BD-02C5,U+02C7-02CC,U+02CE-02D7,U+02DD-02FF,U+1E00-1E9F,U+2020,U+20A0-20AB"}


def main():
    css = []
    for fam, wt, f in FONTS:
        b = base64.b64encode(open(f"{APP}/fonts/{f}.woff2", "rb").read()).decode()
        css.append(f"@font-face{{font-family:'{fam}';font-style:normal;font-weight:{wt};font-display:swap;"
                   f"src:url(data:font/woff2;base64,{b}) format('woff2');unicode-range:{RANGE['ext' if 'ext' in f else 'latin']}}}")
    payload = open(f"{ROOT}/results/ui_payload.json").read().replace("</", "<\\/")
    html = open(f"{APP}/template.html").read().replace("__FONTS__", "\n".join(css)).replace("__PAYLOAD__", payload).replace("__ICONS__", open(f"{APP}/icons.json").read())
    open(f"{APP}/index.html", "w").write(html)
    print("wrote app/index.html", round(len(html) / 1024), "KB")


if __name__ == "__main__":
    main()
