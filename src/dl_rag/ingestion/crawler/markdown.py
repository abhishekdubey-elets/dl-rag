"""HTML → clean-markdown conversion.

Strips scripts/navigation/chrome and boilerplate containers (menus, sidebars,
"related"/"share"/"comment" widgets, breadcrumbs, adverts), then converts the
remaining content to ATX-heading markdown via ``markdownify`` and normalises
whitespace with :func:`dl_rag.utils.text.clean_whitespace`.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

try:  # markdownify exports the ATX constant; fall back to the literal if not.
    from markdownify import ATX, markdownify as _markdownify
except ImportError:  # pragma: no cover - defensive; markdownify is a hard dep
    from markdownify import markdownify as _markdownify  # type: ignore[no-redef]

    ATX = "atx"

from dl_rag.utils.text import clean_whitespace

# Structural elements that never carry article content.
_STRIP_TAGS: tuple[str, ...] = (
    "script",
    "style",
    "noscript",
    "template",
    "iframe",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "svg",
    "button",
)

# class/id substrings that signal boilerplate rather than content.
_BOILERPLATE = re.compile(
    r"(menu|sidebar|side-bar|related|share|sharedaddy|social|comment|breadcrumb"
    r"|advert|widget|newsletter|subscribe|popup|modal|navbar|nav-menu"
    r"|pagination|author-box)",
    re.IGNORECASE,
)


def _identifier(tag: Tag) -> str:
    """Concatenate an element's class list and id for boilerplate matching."""
    classes = tag.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    element_id = tag.get("id") or ""
    return " ".join([*classes, element_id]).strip()


def html_to_markdown(html: str) -> str:
    """Convert an HTML string/fragment to clean ATX markdown.

    Removes chrome and boilerplate, then converts to markdown and normalises
    whitespace. Returns an empty string for empty/whitespace-only input.
    """
    if not html or not html.strip():
        return ""

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()

    # Remove containers whose class/id looks like boilerplate. Iterate over a
    # static snapshot and guard against already-detached nodes.
    for element in list(soup.find_all(True)):
        if element.decomposed:  # type: ignore[attr-defined]
            continue
        try:
            identifier = _identifier(element)
            if identifier and _BOILERPLATE.search(identifier):
                element.decompose()
        except Exception:  # noqa: BLE001 - never let one node break conversion
            continue

    try:
        markdown = _markdownify(str(soup), heading_style=ATX, strip=["a"])
    except Exception:  # noqa: BLE001 - fall back to plain text on md failure
        markdown = soup.get_text("\n")

    return clean_whitespace(markdown)
