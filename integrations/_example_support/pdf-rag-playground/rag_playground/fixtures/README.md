# Public demonstration fixture

`field-guide.pdf` is original fictional text written for this playground. It is
redistributed under the repository's MIT license. It describes an invented Atlas
incident, maintenance policy, and release checklist. No real customer documents,
credentials, or operational data are included.

`questions.json` contains five reviewed questions and reference answers, including
one unanswerable question. Supporting labels include extracted page numbers and
inclusive-start/exclusive-end Python-string offsets. Questions and labels are
frozen before comparative runs. They are demonstration cases, not a benchmark or
evidence of general superiority. Keep failed answers in exported fixture runs.

Rebuild intentionally with `python scripts/build_fixture.py` from the playground
directory, using the pinned `reportlab` development dependency. The PDF bytes are
deterministic with that generator/version. Do not regenerate labels after looking
at model answers to improve a score.
