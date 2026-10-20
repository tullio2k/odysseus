"""The cached-model scan must read config.json to find vision checkpoints.

The name marker catches 33.9% of them (81 of the 239 vision models measured
below; 84/260 against the catalogs), and a model scanned off disk carries no
`pipeline_tag` at all. So a VLM already downloaded — the common case for Serve —
was classified as text and launched with `python -m mlx_lm.server`, which
accepts no images: the server comes up healthy with no way to hand it a picture.

The signal that is actually on disk is the checkpoint's own config.json, where a
vision-language model keeps its vision tower. Measured by laying 483 real
config.json files out as an HF cache and running this scanner over them: the
rule fires on 238/239 of the ones the catalogs tag image-text-to-text, and on 10
of the 244 tagged text-generation — those ten being catalog tag errors rather
than detector errors (see VISION_CONFIG_KEY_PATTERN in services/hwfit/vision.py).

The scan runs as a standalone script, also over SSH on hosts where the route
cannot see the filesystem, so the read has to happen inside the scanner; the
route only turns the fact into a `modality`.

One thing to know before renaming anything here: a scanned row carries the cache
`path`, and the name marker is matched over it as well as over the repo id. A
test whose name contains "vision", "vl" or "skip-vision" lands in a tmp_path
carrying that token, which flags (or opts out) every row in it and would make
these assertions pass without the code under test. Hence the wording of the test
names, and hence the text control alongside each positive case: same cache path,
opposite verdict.
"""

import json
import os
import subprocess
import sys

from routes.cookbook_helpers import _cached_model_scan_script
from services.hwfit.vision import vision_modality

# Top-level shape of mlx-community/Qwen3.5-35B-A3B-4bit's real config.json, as
# it sits in this machine's HF cache. The name carries no marker and
# `model_type` ("qwen3_5_moe") is in no VLM list, so the tower in config.json is
# the only thing on disk that says this checkpoint takes images. Its upstream,
# Qwen/Qwen3.5-35B-A3B, is tagged image-text-to-text in the catalogs.
QWEN3_5_MOE_VL_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "eos_token_id": [248046, 248044],
    "image_token_id": 248056,
    "model_type": "qwen3_5_moe",
    "text_config": {"hidden_size": 4096, "num_hidden_layers": 48},
    "tie_word_embeddings": False,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 248057,
    "vision_config": {"depth": 27, "hidden_size": 1152, "patch_size": 16},
    "vision_end_token_id": 248054,
    "vision_start_token_id": 248053,
}

# Top-level shape of a real text checkpoint's config.json
# (mlx-community/DR-Venus-4B-RL-mlx-8Bit): no tower anywhere.
TEXT_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 2560,
    "max_position_embeddings": 262144,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "tie_word_embeddings": True,
    "transformers_version": "4.57.1",
    "vocab_size": 151936,
}

# Real top-level fragment of cyankiwi/Ovis2.6-30B-A3B-AWQ-4bit: same thing under
# another name, `vit_config`.
OVIS_VIT_CONFIG = {
    "architectures": ["Ovis2_6_MoeForCausalLM"],
    "model_type": "ovis2_6_moe",
    "hidden_size": 2048,
    "llm_config": {"model_type": "qwen3_moe"},
    "visual_vocab_size": 65536,
    "vit_config": {"model_type": "aimv2", "hidden_size": 1024, "image_size": 448},
}

# Real top-level fragment of mlx-community/gemma-3n-E4B-it-lm-4bit, the
# language-only extraction of a multimodal checkpoint: a scalar named after the
# tower, with no tower behind it. The catalogs tag it text-generation.
GEMMA_3N_LM_CONFIG = {
    "architectures": ["Gemma3nForCausalLM"],
    "model_type": "gemma3n_text",
    "audio_soft_tokens_per_image": 188,
    "image_token_id": 262145,
    "vision_soft_tokens_per_image": 256,
    "hidden_size": 2048,
}

TEXT_CONTROL = "mlx-community/DR-Venus-4B-RL-mlx-8Bit"


def _snapshot(cache, repo_id, config=None, rev="snap1"):
    """Lay out one HF-cache entry the way huggingface_hub does."""
    folder = cache / ("models--" + repo_id.replace("/", "--"))
    snap = folder / "snapshots" / rev
    snap.mkdir(parents=True)
    (folder / "blobs").mkdir(exist_ok=True)
    (folder / "blobs" / "weights.safetensors").write_bytes(b"weights")
    if config is not None:
        payload = config if isinstance(config, str) else json.dumps(config)
        (snap / "config.json").write_text(payload, encoding="utf-8")
    return snap


def _scan(tmp_path, cache=None, model_dirs=()):
    """Run the generated scanner the way /api/model/cached does."""
    home = tmp_path / "empty-home"
    home.mkdir(exist_ok=True)
    script = tmp_path / "scan_cache_config.py"
    script.write_text(
        _cached_model_scan_script([str(d) for d in model_dirs]), encoding="utf-8"
    )
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["HUGGINGFACE_HUB_CACHE"] = str(cache if cache is not None else home / "nohub")
    proc = subprocess.run(
        [sys.executable, str(script)],
        check=True, capture_output=True, text=True, env=env, timeout=120,
    )
    return {m["repo_id"]: m for m in json.loads(proc.stdout)}


def test_scan_flags_a_downloaded_image_model_with_no_marker_in_its_name(tmp_path):
    """The gap: a downloaded VLM whose name says nothing about images."""
    cache = tmp_path / "hub"
    _snapshot(cache, "mlx-community/Qwen3.5-35B-A3B-4bit", QWEN3_5_MOE_VL_CONFIG)
    _snapshot(cache, TEXT_CONTROL, TEXT_CONFIG)

    rows = _scan(tmp_path, cache)

    assert rows["mlx-community/Qwen3.5-35B-A3B-4bit"]["has_vision_config"] is True
    assert vision_modality(rows["mlx-community/Qwen3.5-35B-A3B-4bit"]) == "vision"
    # Same cache path, so this cannot have come from the path.
    assert rows[TEXT_CONTROL]["has_vision_config"] is False
    assert vision_modality(rows[TEXT_CONTROL]) == "text"


def test_scan_leaves_a_text_checkpoint_text(tmp_path):
    """No tower on disk must not become an image row."""
    cache = tmp_path / "hub"
    _snapshot(cache, TEXT_CONTROL, TEXT_CONFIG)
    _snapshot(cache, "unsloth/DeepSeek-R1-GGUF", None)

    rows = _scan(tmp_path, cache)

    assert rows[TEXT_CONTROL]["has_vision_config"] is False
    assert vision_modality(rows[TEXT_CONTROL]) == "text"
    # A checkpoint with no config.json at all is a fact, not an error.
    assert rows["unsloth/DeepSeek-R1-GGUF"]["has_vision_config"] is False
    assert vision_modality(rows["unsloth/DeepSeek-R1-GGUF"]) == "text"


def test_a_tower_under_another_key_still_counts(tmp_path):
    """`vision_config` is the common spelling, not the only one: Ovis names it
    `vit_config`, Phi-4-multimodal `vision_lora`, llava `mm_vision_tower`."""
    cache = tmp_path / "hub"
    _snapshot(cache, "cyankiwi/Ovis2.6-30B-A3B-AWQ-4bit", OVIS_VIT_CONFIG)
    _snapshot(cache, TEXT_CONTROL, TEXT_CONFIG)

    rows = _scan(tmp_path, cache)

    assert rows["cyankiwi/Ovis2.6-30B-A3B-AWQ-4bit"]["has_vision_config"] is True
    assert vision_modality(rows["cyankiwi/Ovis2.6-30B-A3B-AWQ-4bit"]) == "vision"
    assert vision_modality(rows[TEXT_CONTROL]) == "text"


def test_a_leftover_scalar_is_not_a_tower(tmp_path):
    """What makes it a signal is a tower — a sub-config or a model reference —
    not any key named after one. A language-only build keeps the scalars."""
    cache = tmp_path / "hub"
    _snapshot(cache, "mlx-community/gemma-3n-E4B-it-lm-4bit", GEMMA_3N_LM_CONFIG)

    row = _scan(tmp_path, cache)["mlx-community/gemma-3n-E4B-it-lm-4bit"]

    assert row["has_vision_config"] is False
    assert vision_modality(row) == "text"


def test_the_stripped_tower_opt_out_wins_over_the_config_signal(tmp_path):
    """mlx-community/Molmo-7B-D-0924-8bit-skip-vision ships the tower stripped.
    It is a text model however loudly its config.json disagrees."""
    cache = tmp_path / "hub"
    repo = "mlx-community/Molmo-7B-D-0924-8bit-skip-vision"
    _snapshot(cache, repo, QWEN3_5_MOE_VL_CONFIG)

    row = _scan(tmp_path, cache)[repo]

    assert row["has_vision_config"] is True
    assert vision_modality(row) == "text"


def test_an_unreadable_config_does_not_break_the_scan(tmp_path):
    """The scanner also runs over SSH on hosts nobody has inspected. A
    truncated or unparseable config.json must cost that one row's signal, not
    the whole listing."""
    cache = tmp_path / "hub"
    _snapshot(cache, "someone/Truncated-7B", '{"architectures": ["Qwen3For')
    _snapshot(cache, "someone/Not-A-Dict-7B", "[1, 2, 3]")
    _snapshot(cache, "mlx-community/Qwen3.5-35B-A3B-4bit", QWEN3_5_MOE_VL_CONFIG)

    rows = _scan(tmp_path, cache)

    assert rows["someone/Truncated-7B"]["has_vision_config"] is False
    assert rows["someone/Not-A-Dict-7B"]["has_vision_config"] is False
    assert rows["mlx-community/Qwen3.5-35B-A3B-4bit"]["has_vision_config"] is True


def test_only_the_snapshot_root_config_counts(tmp_path):
    """Both a cost rule and a precision rule: one read per snapshot, at the
    root, where a checkpoint keeps its own config. Diffusion repos carry a
    config.json per component, and an image_encoder's has a tower in it while
    the pipeline is not a VLM."""
    cache = tmp_path / "hub"
    snap = _snapshot(cache, "someone/Pipeline-XL", TEXT_CONFIG)
    (snap / "image_encoder").mkdir()
    (snap / "image_encoder" / "config.json").write_text(
        json.dumps(QWEN3_5_MOE_VL_CONFIG), encoding="utf-8"
    )

    row = _scan(tmp_path, cache)["someone/Pipeline-XL"]

    assert row["has_vision_config"] is False
    assert vision_modality(row) == "text"


def test_scan_flags_an_image_model_in_a_local_model_dir(tmp_path):
    """Custom download dirs reach the same Serve list and need the same signal."""
    models = tmp_path / "models"
    local = models / "Qwen3.5-35B-A3B-4bit"
    local.mkdir(parents=True)
    (local / "config.json").write_text(json.dumps(QWEN3_5_MOE_VL_CONFIG), encoding="utf-8")
    (local / "model.safetensors").write_bytes(b"weights")
    control = models / "DR-Venus-4B-RL-mlx-8Bit"
    control.mkdir()
    (control / "config.json").write_text(json.dumps(TEXT_CONFIG), encoding="utf-8")
    (control / "model.safetensors").write_bytes(b"weights")

    rows = _scan(tmp_path, None, model_dirs=[models])

    assert rows["Qwen3.5-35B-A3B-4bit"]["has_vision_config"] is True
    assert vision_modality(rows["Qwen3.5-35B-A3B-4bit"]) == "vision"
    assert vision_modality(rows["DR-Venus-4B-RL-mlx-8Bit"]) == "text"


def test_the_scanner_gets_its_rule_from_the_shared_module():
    """The scanner is a standalone script, generated to run on hosts where
    nothing of this project is installed, so it cannot import the rule. The
    pattern is interpolated into it instead of copied, which is what keeps the
    two from drifting the way the dead engine picker did."""
    from services.hwfit.vision import VISION_CONFIG_KEY_PATTERN

    script = _cached_model_scan_script()

    assert "re.compile(%r, re.IGNORECASE)" % (VISION_CONFIG_KEY_PATTERN,) in script
