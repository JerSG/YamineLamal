# Put the match data here

Gitignored: the match data is licensed for the competition only and must never be committed.

Unpack the bundle you received after enrolment so that these files exist:

```
data/tactical/BAR-ATH_tactical_atd.xml              required (tracking metadata)
data/tactical/BAR-ATH_tactical_atd.txt              required (tracking)
data/tactical/BAR-ATH_tactical_smart_tagging.xml    required
data/broadcast/BAR-ATH_broadcast_atd.xml            used (broadcast fusion)
data/broadcast/BAR-ATH_broadcast_atd.txt            used (broadcast fusion)
data/broadcast/BAR-ATH_broadcast_homography.csv     used (broadcast fusion)
```

The submitted file is produced from these six. The tactical homography and the videos are not used. Without the three broadcast files the pipeline still runs, tactical-only (or pass
`--no-broadcast`). To keep the data elsewhere, set
`MTAH_DATA_DIR=/path/to/bundle` (a folder that contains `tactical/`).
