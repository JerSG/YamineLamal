# Put the match data here

Gitignored: the match data is licensed for the competition only and must never be committed.

Unpack the bundle you received after enrolment so that these files exist:

```
data/tactical/BAR-ATH_tactical_atd.xml              required (tracking metadata)
data/tactical/BAR-ATH_tactical_atd.txt              required for --mode full (tracking)
data/tactical/BAR-ATH_tactical_smart_tagging.xml    required
data/tactical/BAR-ATH_tactical_homography.csv       optional (camera feature)
```

The videos and the broadcast files are not used. To keep the data elsewhere, set
`MTAH_DATA_DIR=/path/to/bundle` (a folder that contains `tactical/`).
