"""Which models are vision (image-text-to-text) models.

Single source of truth on the Python side, used by both paths that produce
model rows for the Cookbook UI:

  * the hwfit catalogs, via models._normalize_model_entry()
  * models scanned off disk, via the /api/model/cached route

It matters because the serve logic has to route vision models to a server that
accepts images. `mlx_lm.server` does not, so a vision checkpoint sent there
comes up looking healthy with no way to pass it a picture.

Three signals, because each is insufficient alone — the first two measured over
the 1553 records in services/hwfit/data/, the third over 483 real config.json
files (see VISION_CONFIG_KEY_PATTERN):

  * `pipeline_tag` marks 260 entries, but it is absent for anything scanned off
    disk and wrong on a few rows (`LFM2-VL-450M-6bit` ships tagged
    text-generation while being a vision-language model).
  * the name marker has near-perfect precision (5 of the 6 rows it flags
    against the tag are genuine VLMs) but only 32.3% recall (84/260): most VLMs
    carry no marker in the name at all, e.g. `gemma-3-27b-it-qat-4bit`.
  * a vision tower in the checkpoint's own config.json is decisive (238/239 on
    the sample behind VISION_CONFIG_KEY_PATTERN), but it exists only for a model
    already downloaded, and only the scanner can read it — on a remote host the
    route never sees that filesystem.

So the rule is their OR, minus an explicit opt-out for checkpoints published
with the vision tower stripped (`...-skip-vision`).

This states what the MODEL declares it can do. Whether this machine can
actually serve it is a separate, runtime question — it depends on mlx-vlm being
present in the launch interpreter — and is deliberately not recorded here.

The JS side keeps a mirror of VISION_NAME_MARKERS in static/js/cookbookVision.js
for model names typed into the serve form by hand, which reach the browser
without passing through either Python path. The two lists are pinned together
by tests/test_cookbook_vision_marker_parity.py.
"""

import json
import os
import re

VISION_PIPELINE_TAG = "image-text-to-text"

# A checkpoint on disk carries no pipeline_tag, and most VLM names carry no
# marker, so for a model already downloaded the strong signal is its own
# config.json: a vision-language model keeps a vision tower in there.
#
# The spelling varies by family, so this matches the class rather than listing
# instances — `vision_config` (Qwen-VL, gemma, Mistral3, Llama 4),
# `vit_config` (Ovis), `vision_lora` (Phi-4-multimodal), `mm_vision_tower`
# (llava), `visual` (Qwen-VL gen 1), `vision_tower`.
#
# The value has to be a sub-config or a model reference, which is what a tower
# looks like. A bare number named after vision is not one: the language-only
# extractions mlx-community/gemma-3n-E4B-it-lm-{4bit,bf16} keep a
# `vision_soft_tokens_per_image` scalar with no tower behind it, and are text.
#
# Measured by laying 483 real config.json files out as an HF cache and running
# the scanner over them — every catalog entry tagged image-text-to-text whose
# config could be fetched (239) plus a matched sample tagged text-generation
# (244). This rule fires on 238/239 of the vision ones against 81/239 (33.9%)
# for the name marker, and on 10 of the 244 text ones.
#
# Those ten look like catalog tag errors rather than detector errors, and for
# most of them the catalogs contradict themselves: QuantTrio/Qwen3.6-{27B,35B-A3B}-AWQ
# are tagged text-generation while cyankiwi's quantizations of the same two
# models are tagged image-text-to-text, and mlx-community/Llama-4-Maverick-17B-16E
# is tagged text-generation while upstream meta-llama/Llama-4-Maverick is not.
# PE-Core-S16-384 is a CLIP-style vision encoder (a text_config and a
# vision_config, no architectures), which is not a text model either. For the
# four Mistral3ForConditionalGeneration rows the only evidence is the config
# itself, which carries a vision_config and a vision_feature_layer.
#
# The one vision model still missed is mlx-community/Molmo-7B-D-0924-bf16, whose
# config.json mentions no tower at all — it is behind trust_remote_code.
#
# Two cheaper-looking signals were measured and dropped: `architectures` ending
# in ForConditionalGeneration adds no recall while matching text-only seq2seq
# models such as T5ForConditionalGeneration, and an `image_token*` key adds
# false positives, among them the text-only plamo-2-1b. `model_type` is
# ambiguous in the catalogs — `lfm2` appears on 7 vision rows and 32 text ones.
VISION_CONFIG_KEY_PATTERN = r"^(vision|visual|vit|mm_vision)(_|$)"

# Tokens that identify a vision-language model by name. Each must appear as a
# whole delimited token, so "navlm" or "revlon" do not match on "vl".
VISION_NAME_MARKERS = (
    "vl",
    "vlm",
    "vision",
    "llava",
    "moondream",
    "internvl",
    "idefics",
    "pixtral",
    "smolvlm",
    "paligemma",
    "minicpm-v",
)

_MARKER_RE = re.compile(
    r"(?:^|[-_./])(" + "|".join(VISION_NAME_MARKERS) + r")(?:$|[-_./0-9])",
    re.IGNORECASE,
)

# Published with the vision tower removed — a text model despite the name and,
# in the catalog, despite the tag.
_SKIP_RE = re.compile(r"skip[-_.]vision", re.IGNORECASE)


def _names(entry):
    """Every field a model row might identify itself by."""
    if not isinstance(entry, dict):
        return ""
    parts = [entry.get("name"), entry.get("repo_id"), entry.get("id"), entry.get("path")]
    return " ".join(str(p) for p in parts if p)


def is_vision_model(entry):
    """True when the model declares image input."""
    if not isinstance(entry, dict):
        return False
    names = _names(entry)
    if _SKIP_RE.search(names):
        return False
    modality = str(entry.get("modality") or "").strip().lower()
    if modality:
        return modality == "vision"
    # Read off the checkpoint's own config.json by the disk scanner. It comes
    # before the tag because it describes this exact checkpoint, while the tag
    # is catalog metadata that is occasionally wrong about it.
    if entry.get("has_vision_config") is True:
        return True
    tag = str(entry.get("pipeline_tag") or "").strip().lower()
    if tag == VISION_PIPELINE_TAG:
        return True
    return bool(_MARKER_RE.search(names))


def vision_modality(entry):
    """"vision" or "text" — the value that goes on the row as `modality`."""
    return "vision" if is_vision_model(entry) else "text"


_VERIFIED_CACHE = {"mtime": None, "models": {}}


def verified_vlm_models():
    """Checkpoints that have actually answered a question about an image here.

    Written only by scripts/smoke_mlx_vlm.py, after a real round trip. Declaring
    image input and surviving a load are not the same thing, and a server that
    came up healthy proves neither — so this stays separate from `modality`,
    which is what the model claims.

    Missing file means nothing has been verified yet, which is not an error.
    """
    try:
        from src.constants import DATA_DIR
        path = os.path.join(DATA_DIR, "mlx_vlm_verified.json")
        mtime = os.path.getmtime(path)
    except Exception:
        _VERIFIED_CACHE.update({"mtime": None, "models": {}})
        return {}
    if _VERIFIED_CACHE["mtime"] != mtime:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            models = data if isinstance(data, dict) else {}
        except Exception:
            models = {}
        _VERIFIED_CACHE.update({"mtime": mtime, "models": models})
    return _VERIFIED_CACHE["models"]
