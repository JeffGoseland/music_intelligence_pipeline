# Data directory

Data layout for the Music Emotion Intelligence Pipeline. The following subdirectories are **git-ignored** (not uploaded):

| Directory | Contents |
|-----------|----------|
| `audio/` | Music files (e.g. DEAM MP3s). One file per track. |
| `deam_csvs/` | DEAM CSV data: `features/` (one CSV per song), `annotations/` (arousal/valence etc.). |
| `processed/` | Pipeline outputs (e.g. `song_features.csv`, `emotion_predictions.csv`). |

Populate by copying or symlinking from your DEAM source (e.g. `/Volumes/Dockcase/Music Research /DEAM`). See project README for pipeline usage.
