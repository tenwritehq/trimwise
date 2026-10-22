"""Create the original MIT-licensed example PDF and freeze source-backed questions."""

from __future__ import annotations

import json
import re
from pathlib import Path

import reportlab
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1] / "rag_playground"
PAGES = (
    (
        "Atlas incident / 08 October",
        [
            (
                "This fictional field guide is a public demonstration fixture for the "
                "Trimwise RAG playground. All names, events, policies, and "
                "measurements below are invented. It is not operational advice."
            ),
            (
                "The morning shift started with routine checks of the frontend, the "
                "network, and the support queue. The network was healthy. No new "
                "frontend deployment had been scheduled. The team copied dashboard "
                "screenshots into the incident folder and compared the traffic pattern "
                "with the previous week."
            ),
            (
                "At 13:55 UTC the Atlas invoice service began returning errors. The "
                "invoice worker retry loop ignored backoff settings and exhausted the "
                "database connection pool. The cause was not a network outage. The "
                "team disabled that retry loop and restarted the affected workers. "
                "Service recovered at 14:32 UTC."
            ),
            (
                "The recovery did not require a rollback of the frontend. Mira owns "
                "the retry regression test. The test must verify that retries respect "
                "backoff settings before the worker is enabled again. The incident "
                "review is scheduled for the next Tuesday; the review date is separate "
                "from the recovery time."
            ),
            (
                "Support staff updated the status page and worked through the pending "
                "invoice queue. They retained the incident log for review. The next "
                "shift should check queue depth, retry behavior, and database "
                "connections, rather than treating a successful homepage check as "
                "proof that invoice processing is healthy."
            ),
        ],
    ),
    (
        "Maintenance / records and backups",
        [
            (
                "The following invented maintenance rules apply only to this fictional "
                "Atlas deployment. They describe backup frequency and retention, not "
                "the behavior of Trimwise or any third-party service."
            ),
            (
                "Backups run every Sunday at 02:00 UTC. Backup snapshots are retained "
                "for 14 days. A restore rehearsal runs on the first Monday of each "
                "month. The rehearsal uses a separate test environment and must not "
                "overwrite the active service database."
            ),
            (
                "The on-call engineer records the snapshot identifier, creation time, "
                "restore result, and any missing records. A successful backup job is "
                "not sufficient evidence that a restore will work. The support queue "
                "and incident-review documents are retained separately from database "
                "snapshots."
            ),
            (
                "Monthly maintenance also includes checking access reviews, expired "
                "credentials, and the list of service owners. Routine documentation "
                "changes do not authorize a production rollout. Any exception to these "
                "fictional rules must be recorded by the service owner before the "
                "maintenance window begins."
            ),
        ],
    ),
    (
        "Rollout / release decision",
        [
            (
                "The release planning meeting covered staffing, translated help pages, "
                "accessibility checks, and the status of staging tests. The support "
                "rota is complete. Those routine planning items do not resolve the "
                "production release blocker."
            ),
            (
                "The production rollout is blocked by an expired signing certificate. "
                "Mira owns certificate renewal. Renewal is due Friday at 16:00 UTC. "
                "The deployment owner must verify the renewed certificate before "
                "approving the rollout; a passing staging test alone is insufficient."
            ),
            (
                "After renewal, the team will check the signing chain, verify the "
                "release artifact, and record approval in the release log. If the "
                "certificate is still expired, the rollout remains blocked. The "
                "checklist deliberately separates who owns the renewal from who "
                "approves the deployment."
            ),
            (
                "The field guide does not specify a quarterly revenue target, customer "
                "account passwords, or the cost of an external API. Questions about "
                "those topics cannot be answered from this document. The appropriate "
                "response is to state that the supplied evidence does not provide the "
                "answer."
            ),
        ],
    ),
)


def main() -> None:
    """Write a readable three-page fixture and source-local supporting span labels."""
    destination = ROOT / "fixtures"
    destination.mkdir(exist_ok=True)
    fonts = Path(reportlab.__file__).parent / "fonts"
    for name, filename in (
        ("FixtureSans", "Vera.ttf"),
        ("FixtureBold", "VeraBd.ttf"),
        ("FixtureItalic", "VeraIt.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "FixtureSans"
    styles["Normal"].fontSize = 10.5
    styles["Normal"].leading = 16
    styles["Title"].fontName = "FixtureBold"
    styles["Heading3"].fontName = "FixtureBold"
    styles["Italic"].fontName = "FixtureItalic"
    styles["Title"].fontSize = 22
    styles["Title"].textColor = colors.HexColor("#153e3a")
    story = []
    for index, (title, paragraphs) in enumerate(PAGES):
        if index:
            story.append(PageBreak())
        story.extend(
            [
                Paragraph("TRIMWISE / RAG LAB", styles["Heading3"]),
                Paragraph(title, styles["Title"]),
                Spacer(1, 15),
            ]
        )
        for paragraph in paragraphs:
            story.extend([Paragraph(paragraph, styles["Normal"]), Spacer(1, 12)])
        story.append(
            Paragraph(
                f"Fictional demonstration fixture - MIT license - Page {index + 1}",
                styles["Italic"],
            )
        )
    pdf = destination / "field-guide.pdf"
    SimpleDocTemplate(
        str(pdf),
        pagesize=A4,
        title="Atlas field guide - fictional RAG fixture",
        author="Trimwise contributors",
        leftMargin=54,
        rightMargin=54,
        topMargin=48,
        bottomMargin=48,
        invariant=1,
    ).build(story)
    extracted = [page.extract_text() for page in PdfReader(pdf).pages]
    cases = [
        (
            "incident",
            "What caused the Atlas outage, and when did service recover?",
            (
                "The invoice worker retry loop ignored backoff settings and exhausted "
                "the database connection pool. Service recovered at 14:32 UTC."
            ),
            1,
            "Service recovered at 14:32 UTC.",
        ),
        (
            "backups",
            "How often do backups run and how long are snapshots retained?",
            ("Backups run every Sunday at 02:00 UTC and snapshots are retained for 14 days."),
            2,
            "Backup snapshots are retained for 14 days.",
        ),
        (
            "rollout",
            ("What blocks the production rollout, who owns renewal, and when is it due?"),
            (
                "An expired signing certificate blocks the rollout. Mira owns renewal, "
                "due Friday at 16:00 UTC."
            ),
            3,
            "Mira owns certificate renewal.",
        ),
        (
            "restore",
            "Does a restore rehearsal overwrite the active service database?",
            (
                "No. It uses a separate test environment and must not overwrite the "
                "active service database."
            ),
            2,
            "The rehearsal uses a separate test environment and must not "
            "overwrite the active service database.",
        ),
        (
            "unanswerable",
            "What is Atlas's quarterly revenue target?",
            "The supplied document does not specify a quarterly revenue target.",
            None,
            None,
        ),
    ]
    frozen = []
    for identity, question, answer, page, quote in cases:
        support = None
        if page is not None and quote is not None:
            match = re.search(
                r"\s+".join(re.escape(word) for word in quote.split()), extracted[page - 1]
            )
            assert match is not None
            support = {
                "page": page,
                "quote": match.group(),
                "start": match.start(),
                "end": match.end(),
            }
        frozen.append(
            {"id": identity, "question": question, "reference_answer": answer, "support": support}
        )
    (destination / "questions.json").write_text(
        json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Created {len(extracted)} pages and froze {len(frozen)} questions before comparison runs."
    )


if __name__ == "__main__":
    main()
