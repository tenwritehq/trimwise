# Position-controlled 160-case review

The original 250-case corpus remains unchanged; the separate 160-case evaluation contains 135 natural cases and 25 controlled relocations.

Source dataset SHA-256: `7b752ea79f19369fa151aa27f8bcfb33abdc48d9a9da609b655640e931c7d428`

## Required manual check

Review the 25 rows marked `controlled_relocation`: their required text is moved unchanged to the end of the context; the source case is excluded from this set.

## Distribution

| Position | Cases |
| --- | ---: |
| beginning | 40 |
| middle | 40 |
| end | 40 |
| multiple | 40 |

| Track | Cases |
| --- | ---: |
| adversarial | 17 |
| evidence_qa | 46 |
| instruction | 26 |
| procedure | 20 |
| real_source | 30 |
| structured | 21 |

## Cases

| Case | Position | Origin | Track | Source type | Derived from |
| --- | --- | --- | --- | --- | --- |
| instruction-05-q1 | beginning | natural | instruction | synthetic_policy |  |
| instruction-06-q1 | beginning | natural | instruction | synthetic_policy |  |
| instruction-07-q1 | beginning | natural | instruction | synthetic_policy |  |
| instruction-08-q1 | beginning | natural | instruction | synthetic_policy |  |
| procedure-05-q1 | beginning | natural | procedure | synthetic_procedure |  |
| procedure-06-q1 | beginning | natural | procedure | synthetic_procedure |  |
| procedure-07-q1 | beginning | natural | procedure | synthetic_procedure |  |
| procedure-08-q1 | beginning | natural | procedure | synthetic_procedure |  |
| real-markdown-it-q1 | beginning | natural | evidence_qa | real_technical_source |  |
| real-public-001 | beginning | natural | real_source | real_code |  |
| real-public-002 | beginning | natural | real_source | real_code |  |
| real-public-003 | beginning | natural | real_source | real_code |  |
| real-public-011 | beginning | natural | real_source | real_api_docs |  |
| real-public-012 | beginning | natural | real_source | real_api_docs |  |
| real-public-014 | beginning | natural | real_source | real_api_docs |  |
| real-public-042 | beginning | natural | real_source | real_web_docs |  |
| real-public-043 | beginning | natural | real_source | real_web_docs |  |
| real-public-044 | beginning | natural | real_source | real_web_docs |  |
| real-public-051 | beginning | natural | real_source | real_research_docs |  |
| real-public-052 | beginning | natural | real_source | real_research_docs |  |
| real-public-053 | beginning | natural | real_source | real_research_docs |  |
| real-public-061 | beginning | natural | real_source | real_data_docs |  |
| real-public-062 | beginning | natural | real_source | real_data_docs |  |
| real-public-063 | beginning | natural | real_source | real_data_docs |  |
| real-public-071 | beginning | natural | real_source | real_security_policy |  |
| real-public-072 | beginning | natural | real_source | real_security_policy |  |
| real-public-073 | beginning | natural | real_source | real_security_policy |  |
| real-public-082 | beginning | natural | real_source | real_specification |  |
| real-public-083 | beginning | natural | real_source | real_specification |  |
| real-public-084 | beginning | natural | real_source | real_specification |  |
| real-python-argparse-q1 | beginning | natural | evidence_qa | real_technical_source |  |
| real-trimwise-ranking-q1 | beginning | natural | evidence_qa | real_technical_source |  |
| real-trimwise-readme-q1 | beginning | natural | evidence_qa | real_technical_source |  |
| structured-03-q1 | beginning | natural | structured | synthetic_structured |  |
| structured-04-q1 | beginning | natural | structured | synthetic_structured |  |
| structured-05-q1 | beginning | natural | structured | synthetic_structured |  |
| synthetic-qa-02-q1 | beginning | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-04-q1 | beginning | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-06-q1 | beginning | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-07-q1 | beginning | natural | evidence_qa | synthetic_incident |  |
| adversarial-01-q1 | middle | natural | adversarial | synthetic_adversarial |  |
| adversarial-02-q1 | middle | natural | adversarial | synthetic_adversarial |  |
| adversarial-03-q1 | middle | natural | adversarial | synthetic_adversarial |  |
| adversarial-03-q2 | middle | natural | adversarial | synthetic_adversarial |  |
| adversarial-04-q1 | middle | natural | adversarial | synthetic_adversarial |  |
| adversarial-04-q2 | middle | natural | adversarial | synthetic_adversarial |  |
| adversarial-05-q1 | middle | natural | adversarial | synthetic_adversarial |  |
| adversarial-05-q2 | middle | natural | adversarial | synthetic_adversarial |  |
| adversarial-06-q1 | middle | natural | adversarial | synthetic_adversarial |  |
| instruction-01-q2 | middle | natural | instruction | synthetic_policy |  |
| instruction-02-q2 | middle | natural | instruction | synthetic_policy |  |
| instruction-03-q2 | middle | natural | instruction | synthetic_policy |  |
| instruction-04-q2 | middle | natural | instruction | synthetic_policy |  |
| instruction-05-q2 | middle | natural | instruction | synthetic_policy |  |
| instruction-06-q2 | middle | natural | instruction | synthetic_policy |  |
| instruction-07-q2 | middle | natural | instruction | synthetic_policy |  |
| instruction-08-q2 | middle | natural | instruction | synthetic_policy |  |
| real-markdown-it-q2 | middle | natural | evidence_qa | real_technical_source |  |
| real-python-argparse-q2 | middle | natural | evidence_qa | real_technical_source |  |
| real-trimwise-ranking-q2 | middle | natural | evidence_qa | real_technical_source |  |
| real-trimwise-readme-q2 | middle | natural | evidence_qa | real_technical_source |  |
| real-trimwise-semantic-q2 | middle | natural | evidence_qa | real_technical_source |  |
| real-trimwise-trimmer-q2 | middle | natural | evidence_qa | real_technical_source |  |
| structured-01-q2 | middle | natural | structured | synthetic_structured |  |
| structured-02-q2 | middle | natural | structured | synthetic_structured |  |
| structured-03-q2 | middle | natural | structured | synthetic_structured |  |
| structured-04-q2 | middle | natural | structured | synthetic_structured |  |
| structured-05-q2 | middle | natural | structured | synthetic_structured |  |
| structured-06-q2 | middle | natural | structured | synthetic_structured |  |
| structured-07-q2 | middle | natural | structured | synthetic_structured |  |
| structured-08-q2 | middle | natural | structured | synthetic_structured |  |
| synthetic-qa-01-q2 | middle | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-03-q2 | middle | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-05-q2 | middle | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-06-q2 | middle | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-07-q2 | middle | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-08-q2 | middle | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-09-q2 | middle | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-10-q2 | middle | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-11-q2 | middle | natural | evidence_qa | synthetic_incident |  |
| adversarial-01-q2-position-end | end | controlled_relocation | adversarial | synthetic_adversarial | adversarial-01-q2 |
| adversarial-02-q2-position-end | end | controlled_relocation | adversarial | synthetic_adversarial | adversarial-02-q2 |
| instruction-01-q1-position-end | end | controlled_relocation | instruction | synthetic_policy | instruction-01-q1 |
| instruction-02-q1-position-end | end | controlled_relocation | instruction | synthetic_policy | instruction-02-q1 |
| instruction-03-q1-position-end | end | controlled_relocation | instruction | synthetic_policy | instruction-03-q1 |
| instruction-04-q1-position-end | end | controlled_relocation | instruction | synthetic_policy | instruction-04-q1 |
| procedure-01-q1-position-end | end | controlled_relocation | procedure | synthetic_procedure | procedure-01-q1 |
| procedure-02-q1-position-end | end | controlled_relocation | procedure | synthetic_procedure | procedure-02-q1 |
| procedure-03-q1-position-end | end | controlled_relocation | procedure | synthetic_procedure | procedure-03-q1 |
| procedure-04-q1-position-end | end | controlled_relocation | procedure | synthetic_procedure | procedure-04-q1 |
| real-markdown-it-q3 | end | natural | evidence_qa | real_technical_source |  |
| real-public-013-position-end | end | controlled_relocation | real_source | real_api_docs | real-public-013 |
| real-public-015-position-end | end | controlled_relocation | real_source | real_api_docs | real-public-015 |
| real-public-030 | end | natural | real_source | real_api_docs |  |
| real-public-032-position-end | end | controlled_relocation | real_source | real_api_docs | real-public-032 |
| real-public-041-position-end | end | controlled_relocation | real_source | real_web_docs | real-public-041 |
| real-public-055-position-end | end | controlled_relocation | real_source | real_research_docs | real-public-055 |
| real-public-066-position-end | end | controlled_relocation | real_source | real_data_docs | real-public-066 |
| real-public-074-position-end | end | controlled_relocation | real_source | real_security_policy | real-public-074 |
| real-public-081-position-end | end | controlled_relocation | real_source | real_specification | real-public-081 |
| real-python-argparse-q3 | end | natural | evidence_qa | real_technical_source |  |
| real-trimwise-ranking-q3 | end | natural | evidence_qa | real_technical_source |  |
| real-trimwise-readme-q3 | end | natural | evidence_qa | real_technical_source |  |
| real-trimwise-semantic-q3 | end | natural | evidence_qa | real_technical_source |  |
| real-trimwise-trimmer-q3 | end | natural | evidence_qa | real_technical_source |  |
| structured-01-q1-position-end | end | controlled_relocation | structured | synthetic_structured | structured-01-q1 |
| structured-01-q3 | end | natural | structured | synthetic_structured |  |
| structured-02-q1-position-end | end | controlled_relocation | structured | synthetic_structured | structured-02-q1 |
| structured-02-q3 | end | natural | structured | synthetic_structured |  |
| structured-03-q3 | end | natural | structured | synthetic_structured |  |
| structured-04-q3 | end | natural | structured | synthetic_structured |  |
| structured-05-q3 | end | natural | structured | synthetic_structured |  |
| structured-06-q3 | end | natural | structured | synthetic_structured |  |
| structured-07-q3 | end | natural | structured | synthetic_structured |  |
| structured-08-q3 | end | natural | structured | synthetic_structured |  |
| synthetic-qa-01-q1-position-end | end | controlled_relocation | evidence_qa | synthetic_incident | synthetic-qa-01-q1 |
| synthetic-qa-02-q2-position-end | end | controlled_relocation | evidence_qa | synthetic_incident | synthetic-qa-02-q2 |
| synthetic-qa-03-q1-position-end | end | controlled_relocation | evidence_qa | synthetic_incident | synthetic-qa-03-q1 |
| synthetic-qa-04-q2-position-end | end | controlled_relocation | evidence_qa | synthetic_incident | synthetic-qa-04-q2 |
| synthetic-qa-05-q1-position-end | end | controlled_relocation | evidence_qa | synthetic_incident | synthetic-qa-05-q1 |
| adversarial-01-q3 | multiple | natural | adversarial | synthetic_adversarial |  |
| adversarial-02-q3 | multiple | natural | adversarial | synthetic_adversarial |  |
| adversarial-03-q3 | multiple | natural | adversarial | synthetic_adversarial |  |
| adversarial-04-q3 | multiple | natural | adversarial | synthetic_adversarial |  |
| adversarial-05-q3 | multiple | natural | adversarial | synthetic_adversarial |  |
| adversarial-06-q3 | multiple | natural | adversarial | synthetic_adversarial |  |
| instruction-01-q3 | multiple | natural | instruction | synthetic_policy |  |
| instruction-02-q3 | multiple | natural | instruction | synthetic_policy |  |
| instruction-03-q3 | multiple | natural | instruction | synthetic_policy |  |
| instruction-04-q3 | multiple | natural | instruction | synthetic_policy |  |
| instruction-05-q3 | multiple | natural | instruction | synthetic_policy |  |
| instruction-06-q3 | multiple | natural | instruction | synthetic_policy |  |
| instruction-07-q3 | multiple | natural | instruction | synthetic_policy |  |
| instruction-08-q3 | multiple | natural | instruction | synthetic_policy |  |
| instruction-09-q3 | multiple | natural | instruction | synthetic_policy |  |
| instruction-10-q3 | multiple | natural | instruction | synthetic_policy |  |
| procedure-01-q2 | multiple | natural | procedure | synthetic_procedure |  |
| procedure-01-q3 | multiple | natural | procedure | synthetic_procedure |  |
| procedure-02-q2 | multiple | natural | procedure | synthetic_procedure |  |
| procedure-02-q3 | multiple | natural | procedure | synthetic_procedure |  |
| procedure-03-q2 | multiple | natural | procedure | synthetic_procedure |  |
| procedure-03-q3 | multiple | natural | procedure | synthetic_procedure |  |
| procedure-04-q2 | multiple | natural | procedure | synthetic_procedure |  |
| procedure-04-q3 | multiple | natural | procedure | synthetic_procedure |  |
| procedure-05-q2 | multiple | natural | procedure | synthetic_procedure |  |
| procedure-05-q3 | multiple | natural | procedure | synthetic_procedure |  |
| procedure-06-q2 | multiple | natural | procedure | synthetic_procedure |  |
| procedure-06-q3 | multiple | natural | procedure | synthetic_procedure |  |
| synthetic-qa-01-q3 | multiple | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-02-q3 | multiple | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-03-q3 | multiple | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-04-q3 | multiple | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-05-q3 | multiple | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-06-q3 | multiple | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-07-q3 | multiple | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-08-q3 | multiple | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-09-q3 | multiple | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-10-q3 | multiple | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-11-q3 | multiple | natural | evidence_qa | synthetic_incident |  |
| synthetic-qa-12-q3 | multiple | natural | evidence_qa | synthetic_incident |  |

## Controlled end variants

| Variant | Source | Original span | New span | Required text preview |
| --- | --- | --- | --- | --- |
| adversarial-01-q2-position-end | adversarial-01-q2 | 4587:4669 | 9282:9364 | EVIDENCE-1: The verified maintenance window is 2026-09-11 from 01:00 to 01:45 UTC. |
| adversarial-02-q2-position-end | adversarial-02-q2 | 4587:4669 | 9282:9364 | EVIDENCE-2: The verified maintenance window is 2026-09-12 from 01:00 to 01:45 UTC. |
| instruction-01-q1-position-end | instruction-01-q1 | 436:542 | 9344:9450 | RULE-1-A: Return exactly three JSON keys: decision, reason, and escalation_required; do not emit Markdown. |
| instruction-02-q1-position-end | instruction-02-q1 | 444:550 | 9250:9356 | RULE-2-A: Return exactly three JSON keys: decision, reason, and escalation_required; do not emit Markdown. |
| instruction-03-q1-position-end | instruction-03-q1 | 461:567 | 9445:9551 | RULE-3-A: Return exactly three JSON keys: decision, reason, and escalation_required; do not emit Markdown. |
| instruction-04-q1-position-end | instruction-04-q1 | 420:526 | 9290:9396 | RULE-4-A: Return exactly three JSON keys: decision, reason, and escalation_required; do not emit Markdown. |
| procedure-01-q1-position-end | procedure-01-q1 | 433:496 | 9009:9072 | EXAMPLE-1-A: symptom 'read-only replica lag' maps to label L1A. |
| procedure-02-q1-position-end | procedure-02-q1 | 412:475 | 9126:9189 | EXAMPLE-2-A: symptom 'read-only replica lag' maps to label L2A. |
| procedure-03-q1-position-end | procedure-03-q1 | 429:492 | 8863:8926 | EXAMPLE-3-A: symptom 'read-only replica lag' maps to label L3A. |
| procedure-04-q1-position-end | procedure-04-q1 | 424:487 | 8944:9007 | EXAMPLE-4-A: symptom 'read-only replica lag' maps to label L4A. |
| real-public-013-position-end | real-public-013 | 102:289 | 4684:4871 | Routes, error handlers, before request, after request, and teardown functions can all be coroutine functions if Flask is |
| real-public-015-position-end | real-public-015 | 73:175 | 16600:16702 | Installing Flask installs the ``flask`` script, a `Click`_ command line interface, in your virtualenv. |
| real-public-032-position-end | real-public-032 | 61:153 | 2673:2765 | The default behavior is to raise a `TimeoutException` after 5 seconds of network inactivity. |
| real-public-041-position-end | real-public-041 | 775:831 | 1630:1686 | Django currently supports two interfaces: WSGI and ASGI. |
| real-public-055-position-end | real-public-055 | 550:617 | 14756:14823 | It removes all features whose variance doesn't meet some threshold. |
| real-public-066-position-end | real-public-066 | 270:329 | 24991:25050 | We recommend using :class:`StringDtype` to store text data. |
| real-public-074-position-end | real-public-074 | 464:601 | 19011:19148 | Input validation should happen as early as possible in the data flow, preferably as soon as the data is received from th |
| real-public-081-position-end | real-public-081 | 711:787 | 4701:4777 | A `Context` is a propagation mechanism which carries execution-scoped values |
| structured-01-q1-position-end | structured-01-q1 | 434:573 | 9528:9667 | ```json { "service": "ledger-1", "timeout_ms": 1501, "retries": 2, "modes": [ "safe", "audit" ], "required": true } ``` |
| structured-02-q1-position-end | structured-02-q1 | 434:573 | 9528:9667 | ```json { "service": "ledger-2", "timeout_ms": 1502, "retries": 3, "modes": [ "safe", "audit" ], "required": true } ``` |
| synthetic-qa-01-q1-position-end | synthetic-qa-01-q1 | 447:618 | 8346:8517 | ### Confirmed trigger For INC-202601, the payment gateway failed because retry storm crossed the guarded threshold at 03 |
| synthetic-qa-02-q2-position-end | synthetic-qa-02-q2 | 4059:4253 | 8340:8534 | ### Recovery action Operators restored the catalog service by enabling bounded worker queue, draining the unsafe backlog |
| synthetic-qa-03-q1-position-end | synthetic-qa-03-q1 | 429:610 | 8466:8647 | ### Confirmed trigger For INC-202603, the identity platform failed because expired signing key crossed the guarded thres |
| synthetic-qa-04-q2-position-end | synthetic-qa-04-q2 | 4060:4253 | 8481:8674 | ### Recovery action Operators restored the analytics pipeline by enabling quarantine topic, draining the unsafe backlog, |
| synthetic-qa-05-q1-position-end | synthetic-qa-05-q1 | 389:568 | 8139:8318 | ### Confirmed trigger For INC-202605, the edge cache failed because stale invalidation token crossed the guarded thresho |
