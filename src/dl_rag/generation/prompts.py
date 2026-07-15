"""Prompt engineering: the analyst persona, per-query-type guidance, and the
message-assembly that turns retrieved chunks + conversation history into the
chat-completion payload.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as _date
from typing import TYPE_CHECKING

from dl_rag.constants import NO_EVIDENCE_MESSAGE
from dl_rag.models.enums import ContentType, QueryType

if TYPE_CHECKING:
    from dl_rag.models.domain import QueryAnalysis, RetrievedChunk

# --------------------------------------------------------------------------- #
# System persona
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = f"""You are a senior education-policy analyst writing for the \
digitalLEARNING archive — a body of reporting, interviews, magazine features and \
policy coverage on Indian education and edtech spanning 2005 to the present.

Answer the user's question with the rigour of a briefing note prepared for \
policymakers.

GROUNDING RULES (non-negotiable):
- Use ONLY the numbered sources provided in the SOURCES block below. Do not draw \
on outside knowledge or anything not present in those sources.
- Support every substantive claim with an inline citation marker such as [1], \
naming the source number(s) you relied on. You may cite several at once, e.g. \
[1][3].
- Never fabricate facts, figures, names, dates, places or quotations.
- EVIDENCE POLICY (graded — do not over-refuse):
  1. If the sources answer the question, answer it fully.
  2. If the sources are RELEVANT but only partially answer it — for example the \
question asks about something current or upcoming while the archive documents \
earlier editions, phases or announcements — DO NOT refuse. Report what the \
archive does document (with citations), and state the boundary plainly, e.g. \
"The archive's most recent coverage of this is X (date); it does not yet contain \
coverage beyond that."
  3. Only when the sources are genuinely irrelevant to the question — none of \
them speak to its subject at all — reply with exactly this sentence and nothing \
else:
"{NO_EVIDENCE_MESSAGE}"
- When the sources disagree, present the conflicting viewpoints explicitly \
rather than smoothing them over.
- TEMPORAL GROUNDING: the user message states today's date. Interpret relative \
words like "next", "upcoming", "latest" and "recent" against that date. An \
announcement whose event date has already passed is PAST coverage — report it \
as the most recent documented edition, not as upcoming, and say plainly when \
the archive contains nothing scheduled beyond it.
- Be concrete and specific: name the people, institutions, policies, schemes, \
states and years involved.

STRUCTURE & STYLE:
- Write in clear, professional markdown using these second-level headings, in \
this order:
  ## Executive Summary
  ## Key Findings
  ## Evidence
  ## Historical Context
  ## Current Situation
  ## Future Outlook
- Finish with a ## Key Takeaways section of concise bullet points.
- Omit a section only when no source speaks to it — never invent content to fill \
a heading.
- Keep the tone measured and analytical: no marketing language and no \
unsupported speculation."""


# --------------------------------------------------------------------------- #
# Per-query-type response guidance
# --------------------------------------------------------------------------- #
QUERY_TYPE_INSTRUCTIONS: dict[QueryType, str] = {
    QueryType.TIMELINE: (
        "Organise the answer strictly in chronological order. Introduce each "
        'year with a fourth-level heading in the form "#### 2020" and keep '
        "everything about that year in a single block. Do not mix events from "
        "different years under the same heading. Order the years from earliest "
        "to latest."
    ),
    QueryType.COMPARISON: (
        "Produce a markdown comparison **table** with one column per item being "
        "compared and rows for the salient dimensions. Follow the table with a "
        "short synthesis paragraph that draws out the key differences and "
        "trade-offs."
    ),
    QueryType.TREND: (
        "Bucket the discussion into these eras: 2005–2010, 2011–2015, "
        "2016–2020, 2021–2023, and 2024–present. For each era, describe what "
        "changed, then close with the overall trajectory and where the trend "
        "appears to be heading."
    ),
    QueryType.DEFINITION: (
        "Lead with a crisp one- or two-sentence definition of the concept, then "
        "expand with its origins, context and how it is applied in the Indian "
        "education landscape."
    ),
    QueryType.RANKING: (
        "Present the answer as an ordered (numbered) list from top to bottom, "
        "and state explicitly the basis for the ranking — the metric or criteria "
        "drawn from the sources."
    ),
    QueryType.STATISTICS: (
        "Foreground the concrete figures — numbers, percentages, counts, dates — "
        "and attach a citation to every figure. Group related statistics "
        "together and note the year each figure refers to."
    ),
    QueryType.POLICY: (
        "Explain the policy or scheme: its objectives, the body that introduced "
        "it, the timeline of rollout, and its reported impact and criticisms."
    ),
    QueryType.INSTITUTION: (
        "Cover the institution's role and mandate, key milestones, leadership, "
        "and notable initiatives as reported in the sources."
    ),
    QueryType.PERSON: (
        "Summarise who the person is, the roles they have held and when, and "
        "their notable statements or actions reported in the sources."
    ),
    QueryType.INTERVIEW: (
        "Draw out the interviewee's key positions and direct quotes (attributed "
        "and cited), along with the context of the conversation."
    ),
    QueryType.MAGAZINE: (
        "Summarise the themes and notable articles of the issue(s), noting the "
        "issue name, month and year where available."
    ),
    QueryType.EVENT: (
        "Cover what happened, when and where, who was involved, and the reported "
        "significance or outcome of the event. For a RECURRING event (e.g. an "
        "annual summit or conference), organise by edition in chronological "
        'order using fourth-level headings such as "#### 27th edition — '
        'Malaysia, 2023", covering venue, participants and outcomes per edition. '
        "If asked about the next or upcoming edition: compare the latest "
        "documented edition's date against the CURRENT DATE. If it is in the "
        "future, present it as upcoming. If it has already passed, say so in "
        "the past tense and state plainly that the archive does not yet contain "
        "an announcement for a later edition."
    ),
    QueryType.RECOMMENDATION: (
        "Give a reasoned recommendation grounded strictly in the evidence: lay "
        "out the options and their trade-offs before stating the suggested "
        "course of action."
    ),
    QueryType.SUMMARIZATION: (
        "Provide a faithful, well-structured summary of the source material, "
        "preserving the relative emphasis of the originals."
    ),
    QueryType.GENERAL: (
        "Answer directly and analytically, following the standard section "
        "structure and grounding every claim in the numbered sources."
    ),
}


def query_type_instruction(query_type: QueryType) -> str:
    """Return the guidance for ``query_type`` with a GENERAL fallback."""
    return QUERY_TYPE_INSTRUCTIONS.get(
        query_type, QUERY_TYPE_INSTRUCTIONS[QueryType.GENERAL]
    )


# --------------------------------------------------------------------------- #
# Context rendering
# --------------------------------------------------------------------------- #
def _relative_age(published: _date | None) -> str:
    """Human phrase for how long ago a source was published, e.g. '2 years ago'.

    Computed in code so the LLM never has to do date arithmetic — small models
    reliably follow "2 years ago" but not "compare 2024-07-03 with today".
    """
    if published is None:
        return ""
    days = (_date.today() - published).days
    if days < 0:
        return "dated in the future"
    if days == 0:
        return "published today"
    if days < 60:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 730:
        months = days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    return f"{days // 365} years ago"


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render chunks as numbered source blocks aligned with citation indices.

    Each block looks like::

        [1] Title (content_type, June 2024, 2 years ago) — https://...
        <chunk text>
    """
    blocks: list[str] = []
    for index, retrieved in enumerate(chunks, start=1):
        meta = retrieved.chunk.metadata
        content_type = (
            meta.content_type.value
            if isinstance(meta.content_type, ContentType)
            else str(meta.content_type)
        )
        when = " ".join(str(part) for part in (meta.month, meta.year) if part)
        age = _relative_age(meta.published_date)
        parts = [part for part in (content_type, when, age) if part]
        header = f"[{index}] {meta.title} ({', '.join(parts)}) — {meta.url}"
        blocks.append(f"{header}\n{retrieved.chunk.text.strip()}\n")
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# Message assembly
# --------------------------------------------------------------------------- #
def build_messages(
    analysis: QueryAnalysis,
    chunks: Sequence[RetrievedChunk],
    history_summary: str | None = None,
    history_turns: list[dict[str, str]] | None = None,
    no_evidence: bool = False,
) -> list[dict[str, str]]:
    """Compose the chat-completion messages for an answer-generation pass."""
    today = _date.today().isoformat()
    system = (
        f"CURRENT DATE: {today}. Every source is dated; any event dated before "
        f"{today} has ALREADY HAPPENED and must be described in the past tense — "
        "regardless of the (future) tense its announcement article used.\n\n"
        f"{SYSTEM_PROMPT}\n\n## Response format for this query\n"
        f"{query_type_instruction(analysis.query_type)}"
    )
    if ContentType.VIDEO in analysis.content_type_filter:
        system += (
            "\n\nThe user is asking for VIDEOS. Skip the standard section "
            "structure. Reply with a short lead sentence followed by a markdown "
            "bulleted list — one bullet per video: **title** (date, duration if "
            "known), a one-line description of what it covers, and the citation "
            "marker [n]. The numbered sources ARE the video links; list every "
            "distinct relevant video from the sources and nothing else."
        )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    if history_summary and history_summary.strip():
        messages.append(
            {
                "role": "system",
                "content": (
                    "Summary of the earlier conversation (for context; still cite "
                    f"only the numbered sources):\n{history_summary.strip()}"
                ),
            }
        )

    for turn in history_turns or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    if no_evidence or not chunks:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question: {analysis.original_query}\n\n"
                    "There are no usable sources for this question. Reply with "
                    f"exactly the following sentence and nothing else:\n"
                    f"{NO_EVIDENCE_MESSAGE}"
                ),
            }
        )
        return messages

    messages.append(
        {
            "role": "user",
            "content": (
                f"Today's date: {_date.today().isoformat()}\n\n"
                f"SOURCES:\n{format_context(chunks)}\n\n"
                f"Question: {analysis.original_query}\n\n"
                "Write your answer using ONLY the numbered sources above, with "
                "inline [n] citations for every claim."
            ),
        }
    )
    return messages
