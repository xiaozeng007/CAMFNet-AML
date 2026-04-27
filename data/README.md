# Data Preparation (Anonymous Release)

This repository does not include raw datasets or private checkpoints.

## Expected Layout

```text
data/
  processed/
    MUStARD++/
    MOSEI/
```

## Required Inputs

- Pre-extracted multimodal features (text/audio/vision) in pickle format.
- Dataset split metadata compatible with config files under `cafnet_aml/config/`.

## Notes

- Do not commit raw videos, waveforms, or private metadata.
- Keep paths relative to repository root.
- Update `dataset_root_dir` and `featurePath*` fields in config JSON files before running scripts.
