# Generated output

`python run.py --team <team_name>` writes two files here:

| file | what |
|---|---|
| `<team_name>.csv` | the possession file, 199,319 rows: upload it through the submission form |
| `<team_name>_report.json` | the model used (strict-external), ball-fusion statistics and output statistics |

Both are gitignored. A possession file is a full set of labels for the match, so it is submitted
through the form and never published in the repository.
