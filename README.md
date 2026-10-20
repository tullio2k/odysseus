# Screenshots for PR #6771

These two images are referenced from the description of
odysseus-dev/odysseus#6771, which changes what the Cookbook model list renders
and therefore needs visual evidence per the repository's PR template.

They live on a branch of their own, outside the pull request's diff, so that
reviewing the change does not mean reviewing two binaries. Nothing in the
codebase references them.

- `cookbook-vlm-badge-list.jpg` — the model list. Rows whose model declares
  image input carry a `VLM` badge and read `MLX VLM` in the Mode column; the
  text models beside them still read `MLX`.
- `cookbook-vlm-panel-note.jpg` — the expanded serve panel for
  `mlx-community/gemma-3-27b-it-qat-bf16`, showing the note that explains why
  the model needs `mlx_vlm.server` rather than `mlx_lm.server`.

Captured from the app running locally (`uvicorn app:app`) on an Apple M4 Max.
