### Structure

```
forest-sounds/
  ├── data/      ← drop dataset here
  ├── models/    ← save checkpoints
  ├── figures/   ← export plots
  ├── utils/     ← helper modules
  ├── sound_classification_report.ipynb
  └── soundedge-esc-application   ← huggingface space + added training scripts
```

[PasanSarathchandra/soundedge-esc-application](https://huggingface.co/spaces/PasanSarathchandra/soundedge-esc-application/tree/main)

### Utils

```
  ┌─────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
  │    File     │                                      Contents                                       │
  ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ io.py       │ load_fixed, list_audio, build_metadata, normalize                                   │
  ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ features.py │ mel_spectrogram, log_mel, mfcc, extract_handcrafted, batch_features, normalize_spec │
  ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ augment.py  │ aug_noise/time_shift/pitch/time_stretch/gain, spec_augment, random_augment          │
  ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ viz.py      │ plot_waveform/spectrogram/features/confusion/training_curves/class_distribution     │
  ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ data.py     │ AudioDataset (torch), make_splits, encode_labels                                    │
  ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ train.py    │ fit, train_epoch, eval_epoch, predict, EarlyStopping                                │
  ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ metrics.py  │ classification_metrics, per_class_f1, inference_latency, count_params               │
  ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ __init__.py │ re-exports all                                                                      │
  └─────────────┴─────────────────────────────────────────────────────────────────────────────────────┘
```
