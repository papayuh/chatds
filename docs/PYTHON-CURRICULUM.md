# Python curriculum tools

`corpus/python_tasks.py` defines reference tasks. `offline_python.py` generates
small deterministic phrasing variations; `python_curriculum.py` and
`curriculum_v2.py` produce larger project-owned Python/bugfix families.
They enforce answer syntax/budget checks and screen frozen evaluation overlap.
They do not execute generated answers.

```sh
python3 corpus/curriculum_v2.py --check-suite
python3 corpus/curriculum_v2.py --out-dir build/curriculum
python3 corpus/offline_python.py --out build/offline-python.jsonl
```

Holdouts and training sets must remain separate. Report family/category scores,
not only pooled accuracy; template correctness is not unseen-task competence.
Use `eval/audit_python.py`, `eval/audit_autoresearch.py` and the diagnostics in
`eval/` for whole-answer syntax/binding and budget checks.

The historical automated search controller and end-to-end campaign shell
wrappers required a removed runtime planner/exporter; they are not shipped.
Use the standalone generators, trainer and clean host evaluator directly.
See [training](../train/README.md) and [offline data](OFFLINE-DATA.md).
