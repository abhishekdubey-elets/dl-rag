"""Domain constants: content-type routing, a curated entity gazetteer, and the
canonical relation vocabulary used by the knowledge graph.

The gazetteer is deliberately hand-curated for the Indian education / edtech
domain that digitalLEARNING covers. It makes entity extraction *deterministic*
and high-precision for the entities that matter most (regulators, policies,
schemes, marquee institutions), and is supplemented at runtime by optional
statistical NER for the long tail.
"""

from __future__ import annotations

from dl_rag.models.enums import EntityType

# --- Mapping of digitalLEARNING URL/category slugs to internal content types ---
CATEGORY_SLUG_TO_CONTENT_TYPE: dict[str, str] = {
    "news": "news",
    "interview": "interview",
    "interviews": "interview",
    "article": "magazine_article",
    "articles": "magazine_article",
    "magazine": "magazine_issue",
    "cover-story": "cover_story",
    "special-story": "special_story",
    "policy": "policy",
    "policy-matters": "policy",
    "higher-education": "higher_education",
    "school-education": "school",
    "schools": "school",
    "government": "government_news",
    "governance": "government_news",
    "corporate": "corporate",
    "features": "feature",
    "feature": "feature",
    "opinion": "opinion",
    "rankings": "ranking",
    "ranking": "ranking",
}

# --- Curated entity gazetteer: canonical name -> (EntityType, aliases) ----------
# Aliases are matched case-insensitively as whole tokens/phrases.
GAZETTEER: dict[str, tuple[EntityType, list[str]]] = {
    # Regulators & bodies
    "UGC": (EntityType.REGULATOR, ["University Grants Commission"]),
    "AICTE": (EntityType.REGULATOR, ["All India Council for Technical Education"]),
    "NAAC": (EntityType.REGULATOR, ["National Assessment and Accreditation Council"]),
    "NIRF": (EntityType.REGULATOR, ["National Institutional Ranking Framework"]),
    "NBA": (EntityType.REGULATOR, ["National Board of Accreditation"]),
    "CBSE": (EntityType.REGULATOR, ["Central Board of Secondary Education"]),
    "NCERT": (EntityType.REGULATOR, ["National Council of Educational Research and Training"]),
    "NCTE": (EntityType.REGULATOR, ["National Council for Teacher Education"]),
    "NTA": (EntityType.REGULATOR, ["National Testing Agency"]),
    "NCVET": (EntityType.REGULATOR, ["National Council for Vocational Education and Training"]),
    # Ministries / departments
    "Ministry of Education": (
        EntityType.GOVERNMENT_DEPARTMENT,
        ["MoE", "MHRD", "Ministry of Human Resource Development"],
    ),
    "Department of School Education and Literacy": (EntityType.GOVERNMENT_DEPARTMENT, ["DoSEL"]),
    "Department of Higher Education": (EntityType.GOVERNMENT_DEPARTMENT, ["DHE"]),
    # Policies & schemes
    "NEP 2020": (
        EntityType.POLICY,
        ["National Education Policy 2020", "National Education Policy", "NEP"],
    ),
    "RTE Act": (EntityType.POLICY, ["Right to Education Act", "Right to Education"]),
    "SWAYAM": (EntityType.SCHEME, ["Study Webs of Active Learning for Young Aspiring Minds"]),
    "SWAYAM Prabha": (EntityType.SCHEME, []),
    "DIKSHA": (EntityType.SCHEME, ["Digital Infrastructure for Knowledge Sharing"]),
    "PM eVIDYA": (EntityType.SCHEME, ["PM e-VIDYA", "PMeVIDYA"]),
    "Samagra Shiksha": (EntityType.SCHEME, ["Samagra Shiksha Abhiyan"]),
    "Sarva Shiksha Abhiyan": (EntityType.SCHEME, ["SSA"]),
    "RUSA": (EntityType.SCHEME, ["Rashtriya Uchchatar Shiksha Abhiyan"]),
    "NPTEL": (EntityType.SCHEME, ["National Programme on Technology Enhanced Learning"]),
    "NDLI": (EntityType.SCHEME, ["National Digital Library of India"]),
    "APAAR": (EntityType.SCHEME, ["Automated Permanent Academic Account Registry"]),
    "PARAKH": (EntityType.SCHEME, []),
    "ULLAS": (EntityType.SCHEME, ["New India Literacy Programme"]),
    "NISHTHA": (EntityType.SCHEME, []),
    # Marquee institutions (representative; the long tail is caught by NER + fuzzy match)
    "IIT Madras": (EntityType.UNIVERSITY, ["Indian Institute of Technology Madras", "IIT-M"]),
    "IIT Delhi": (EntityType.UNIVERSITY, ["Indian Institute of Technology Delhi"]),
    "IIT Bombay": (EntityType.UNIVERSITY, ["Indian Institute of Technology Bombay"]),
    "IIT Kanpur": (EntityType.UNIVERSITY, ["Indian Institute of Technology Kanpur"]),
    "IIT Kharagpur": (EntityType.UNIVERSITY, ["Indian Institute of Technology Kharagpur"]),
    "IISc Bangalore": (EntityType.UNIVERSITY, ["Indian Institute of Science"]),
    "JNU": (EntityType.UNIVERSITY, ["Jawaharlal Nehru University"]),
    "Delhi University": (EntityType.UNIVERSITY, ["University of Delhi", "DU"]),
    "IGNOU": (EntityType.UNIVERSITY, ["Indira Gandhi National Open University"]),
    "Ashoka University": (EntityType.UNIVERSITY, []),
    "Amity University": (EntityType.UNIVERSITY, []),
    # Elets flagship events (heavily covered by the archive itself)
    "World Education Summit": (
        EntityType.EVENT,
        ["WES", "Elets World Education Summit", "Elets WES"],
    ),
    "Higher Education & Human Resource Conclave": (EntityType.EVENT, ["HEHR Conclave"]),
    "School Leadership Summit": (EntityType.EVENT, ["Elets School Leadership Summit"]),
    # EdTech companies / products
    "BYJU'S": (EntityType.EDTECH_STARTUP, ["Byjus", "Byju's"]),
    "Unacademy": (EntityType.EDTECH_STARTUP, []),
    "upGrad": (EntityType.EDTECH_STARTUP, []),
    "PhysicsWallah": (EntityType.EDTECH_STARTUP, ["Physics Wallah", "PW"]),
    "Vedantu": (EntityType.EDTECH_STARTUP, []),
    "Coursera": (EntityType.EDTECH_STARTUP, []),
    "Google": (EntityType.COMPANY, ["Google for Education"]),
    "Microsoft": (EntityType.COMPANY, ["Microsoft Education"]),
    "AWS": (EntityType.COMPANY, ["Amazon Web Services"]),
}

# --- Indian states / UTs (for time/place filtering and KG place nodes) ---------
INDIAN_STATES: list[str] = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh",
]

# --- Canonical relation phrases the KG extractor looks for between entities ----
# predicate -> trigger phrases (lowercase)
RELATION_TRIGGERS: dict[str, list[str]] = {
    "introduced": ["introduced", "launched", "unveiled", "rolled out", "notified"],
    "partnered_with": ["partnered with", "in partnership with", "tie-up with",
                        "collaborated with", "signed mou with", "joined hands with"],
    "launched": ["launched", "inaugurated", "set up", "established"],
    "implemented": ["implemented", "adopted", "rolled out", "deployed"],
    "leads": ["led by", "headed by", "chaired by", "vice chancellor of",
              "director of", "appointed as"],
    # Person-gated in the extractor: only emitted when the subject is a PERSON
    # entity (from NER), so "UGC addressed concerns…" never becomes a relation.
    "spoke_at": ["spoke at", "speaking at", "delivered the keynote",
                 "delivered a keynote", "keynote address", "keynote speaker",
                 "delivered the inaugural address", "addressed the",
                 "panelist at", "joined the panel", "graced the occasion",
                 "chief guest at", "shared insights at", "shared his views",
                 "shared her views"],
    "located_in": ["based in", "located in", "in the state of", "headquartered in"],
    "part_of": ["part of", "under the", "affiliated to", "affiliated with"],
}

# Query intent keyword hints (used alongside the LLM/heuristic classifier).
INTENT_KEYWORDS: dict[str, list[str]] = {
    "timeline": ["timeline", "evolution", "over the years", "history of",
                 "since", "chronology", "developments"],
    "comparison": ["compare", "comparison", "versus", "vs", "difference between",
                   "against"],
    "trend": ["trend", "trends", "how has", "changed over", "growth", "shift",
              "emerging"],
    "definition": ["what is", "what are", "define", "meaning of", "explain"],
    "ranking": ["top", "best", "ranking", "rank", "leading", "which states"],
    "interview": ["interview", "interviews", "in conversation", "featuring"],
    "person": ["who is", "vice chancellor", "minister", "ceo", "founder"],
    "statistics": ["how many", "number of", "percentage", "statistics", "figures"],
    "recommendation": ["recommend", "suggest", "which should", "best option"],
}

CITATION_MARKER = "[{index}]"
NO_EVIDENCE_MESSAGE = (
    "I could not find enough supporting evidence in the digitalLEARNING archive "
    "to answer this confidently."
)
