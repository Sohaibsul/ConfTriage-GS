"""Convert conftriage_gs.py (# %% cell markers) into a Kaggle .ipynb and write kernel-metadata.json.
Usage: python make_ipynb.py <kaggle_username>"""
import json, sys, re, pathlib

here = pathlib.Path(__file__).parent
user = sys.argv[1] if len(sys.argv) > 1 else "YOUR_KAGGLE_USERNAME"
slug = "conftriage-gs"
src = (here / "conftriage_gs.py").read_text(encoding="utf-8")

cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    is_md = block.startswith(" [markdown]")
    body = block.split("\n", 1)[1].strip("\n") if "\n" in block else ""
    if is_md:
        body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": body})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body})

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU"}}
(here / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

meta = {
    "id": f"{user}/{slug}",
    "title": "ConfTriage-GS",
    "code_file": f"{slug}.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
(here / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"wrote {slug}.ipynb ({len(cells)} cells) and kernel-metadata.json for '{user}'")
