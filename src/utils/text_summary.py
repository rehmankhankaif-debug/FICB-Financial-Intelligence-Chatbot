from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence


MAX_SENTENCE_CHARS = 280
DEFAULT_SUMMARY_POINTS = 5

SECTION_HEADINGS = (
    "Objective",
    "Data Handling",
    "Document Processing",
    "Financial Data Analysis",
    "User Query Support",
    "Intent Recognition",
    "Tool Selection",
    "Coupling",
    "Expandability",
    "User Interface",
    "Frontend",
    "Features",
    "Backend",
    "Evaluation",
    "Deliverables",
)

_BULLET_RE = re.compile(r"[\u00b7\u2022\u2023\u2043\u2219\u25a0-\u25cf\u25e6\uf0b7]+")
_HEADING_RE = re.compile(r"\b({0})\b\s*:?\s*".format("|".join(re.escape(item) for item in SECTION_HEADINGS)), re.IGNORECASE)
_HEADING_SPLIT_RE = re.compile(r"\s+(?=(?:{0}):\s)".format("|".join(re.escape(item) for item in SECTION_HEADINGS)), re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9/%$.,+-]*")
_INCOMPLETE_ENDINGS = {"and", "or", "with", "between", "for", "to", "of", "in", "on", "using", "as", "when", "while", "including"}
_DECORATIVE_RUN_RE = re.compile(r"(?<![A-Za-z0-9])[\-_=*~#|:.]{4,}(?![A-Za-z0-9])")
_DECORATIVE_LINE_RE = re.compile(r"^[\s\-_=*~#|:.,`'\"/\\]+$")


def is_decorative_line(line: str) -> bool:
    compact = re.sub(r"\s+", "", str(line or ""))
    if len(compact) < 3:
        return False
    if any(char.isalnum() for char in compact):
        return False
    return bool(_DECORATIVE_LINE_RE.fullmatch(compact))


def clean_document_text(text: str) -> str:
    raw_text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: List[str] = []
    for line in raw_text.splitlines():
        if is_decorative_line(line):
            continue
        cleaned = _DECORATIVE_RUN_RE.sub(". ", line)
        cleaned = re.sub(r"(?:\.\s*){2,}", ". ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        if cleaned and not is_decorative_line(cleaned):
            cleaned_lines.append(cleaned)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"(?:\.\s*){2,}", ". ", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    return cleaned_text.strip()


def normalize_extractive_text(text: str) -> str:
    cleaned = clean_document_text(text).replace("\r", " ").replace("\n", " ")
    cleaned = _BULLET_RE.sub(". ", cleaned)
    cleaned = re.sub(r"\bsummari\b", "summaries", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _HEADING_RE.sub(lambda match: _heading_boundary(match, cleaned), cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r":\s*\.\s*", ": ", cleaned)
    cleaned = re.sub(r"([.!?]){2,}", r"\1", cleaned)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    return cleaned.strip(" .")


def extract_candidate_sentences(texts: Iterable[str], limit: int = 12) -> List[str]:
    candidates: List[str] = []
    seen = set()
    for text in texts:
        prepared = normalize_extractive_text(text)
        if not prepared:
            continue
        for unit in _split_units(prepared):
            candidate = _clean_unit(unit)
            if not _is_substantive(candidate):
                continue
            candidate = _trim_to_sentence(candidate)
            key = _dedupe_key(candidate)
            if not key or key in seen or _is_near_duplicate(key, seen):
                continue
            seen.add(key)
            candidates.append(candidate)
            if len(candidates) >= limit:
                return candidates
    return candidates


def build_extractive_answer(texts: Iterable[str], max_sentences: int = 3) -> str:
    candidates = extract_candidate_sentences(texts, limit=max_sentences)
    return combine_sentences(candidates, limit=720)


def build_extractive_summary(texts: Iterable[str], mode: str = "summary", max_points: int = DEFAULT_SUMMARY_POINTS) -> str:
    text_list = [str(text or "") for text in texts]
    key_fact_summary = build_key_fact_statement_summary(text_list)
    if key_fact_summary:
        return key_fact_summary

    candidates = extract_candidate_sentences(text_list, limit=max_points + 3)
    if not candidates:
        return ""

    if mode == "outline":
        return "\n".join("{0}. {1}".format(index + 1, point) for index, point in enumerate(candidates[:max_points]))
    if mode == "key_points":
        return "\n".join("- {0}".format(point) for point in candidates[:max_points])
    if mode == "tldr":
        return combine_sentences(candidates[:2], limit=420)

    intro = combine_sentences(candidates[:2], limit=520)
    remaining = candidates[2 : 2 + max_points]
    if not remaining:
        return intro

    heading = "Executive summary" if mode == "executive_summary" else "Summary"
    bullets = "\n".join("- {0}".format(point) for point in remaining)
    return "{0}: {1}\n\nKey points:\n{2}".format(heading, intro, bullets)


def combine_sentences(sentences: Sequence[str], limit: int = 700) -> str:
    combined = " ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip())
    if len(combined) <= limit:
        return combined
    return _trim_to_sentence(combined, limit=limit)


def build_key_fact_statement_summary(texts: Iterable[str]) -> str:
    text = _plain_text(" ".join(str(item or "") for item in texts))
    lower_text = text.lower()
    if "key fact statement" not in lower_text or "sanctioned loan amount" not in lower_text:
        return ""

    fields = _extract_kfs_fields(text)
    if not fields:
        return ""

    lead_parts = []
    if fields.get("date"):
        lead_parts.append("dated {0}".format(fields["date"]))
    if fields.get("loan_type"):
        lead_parts.append("for {0}".format(fields["loan_type"]))
    if fields.get("reference"):
        lead_parts.append("reference {0}".format(fields["reference"]))

    lead = "This Key Fact Statement"
    if lead_parts:
        lead = "{0} {1}".format(lead, ", ".join(lead_parts))

    amount_term = []
    if fields.get("sanctioned_amount"):
        amount_term.append("sanctioned amount {0}".format(fields["sanctioned_amount"]))
    if fields.get("loan_term"):
        amount_term.append("loan term {0}".format(fields["loan_term"]))
    if amount_term:
        lead = "{0} covers {1}".format(lead, " and ".join(amount_term))

    if fields.get("tranche_month") and fields.get("tranche_amount"):
        lead = "{0}; Tranche-1 is scheduled for {1} with {2}".format(lead, fields["tranche_month"], fields["tranche_amount"])
    lead = _sentence(lead)

    points = []
    repayment = _join_existing(_repayment_phrases(fields))
    if repayment:
        points.append("Repayment: {0}.".format(repayment))

    interest = _join_existing(
        [
            fields.get("interest_rate"),
            _field_phrase("total borrower interest", fields.get("total_interest")),
        ]
    )
    if interest:
        points.append("Interest: {0}.".format(interest))

    charges = _join_existing(
        [
            _field_phrase("APR", fields.get("apr")),
            _zero_fee_phrase(fields),
        ]
    )
    if charges:
        points.append("APR and fees: {0}.".format(charges))

    contingent = _join_existing(
        [
            _field_phrase("delayed-payment penal charge", fields.get("penal_charge")),
            _field_phrase("foreclosure charge", fields.get("foreclosure_charge")),
            _field_phrase("part-payment charge", fields.get("part_payment_charge")),
        ]
    )
    if contingent:
        points.append("Contingent charges: {0}.".format(contingent))

    if fields.get("validity"):
        points.append("Validity: KFS is valid for {0}.".format(fields["validity"]))

    if not points:
        return lead
    return "Summary: {0}\n\nKey points:\n{1}".format(lead, "\n".join("- {0}".format(point) for point in points[:5]))


def _heading_boundary(match: re.Match[str], full_text: str) -> str:
    heading = match.group(1)
    prefix = "" if match.start() == 0 else ". "
    return "{0}{1}: ".format(prefix, heading)


def _plain_text(text: str) -> str:
    cleaned = clean_document_text(text).replace("\r", " ").replace("\n", " ")
    cleaned = _BULLET_RE.sub(" ", cleaned)
    cleaned = re.sub(r"Ver-[A-Za-z’']+\d+\s+Page\s+\d+\s+of\s+\d+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_]{4,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_kfs_fields(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    fields["reference"] = _first_match(text, r"\bRef No:\s*([A-Za-z0-9/-]+)") or _first_match(text, r"Loan proposal/ account No\.?\s*([A-Za-z0-9/-]+)")
    fields["date"] = _first_match(text, r"\bDate:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})")
    fields["loan_type"] = _clean_value(_first_match(text, r"Type of Loan\s+(.+?)\s+\d+\s+Sanctioned Loan amount"))
    fields["sanctioned_amount"] = _money(_first_match(text, r"Sanctioned Loan amount.*?\b(?:INR|Rs\.?)\s*([0-9,]+)"), "INR")
    fields["tranche_month"] = _first_match(text, r"Tranche-1\s+([A-Za-z]+\s+\d{4})\s+[0-9,]+")
    fields["tranche_amount"] = _money(_first_match(text, r"Tranche-1\s+[A-Za-z]+\s+\d{4}\s+([0-9,]+)"), "INR")
    fields["loan_term"] = _first_match(text, r"Loan term\s*\(year/months/days\)\s*([0-9]+\s+[A-Za-z]+)")
    fields["installment_frequency"] = _first_match(text, r"post sanction\s+([A-Za-z]+)\s+[0-9\s-]+\s+\(EMI\)")
    fields["epi_range"] = _first_match(text, r"post sanction\s+[A-Za-z]+\s+([0-9]+\s*-\s*[0-9]+)\s+\(EMI\)")
    fields["emi"] = _money(_first_match(text, r"\(EMI\)\s+(?:INR|Rs\.?)\s*([0-9,]+)\s+\(EMI\)"), "INR")
    fields["repayment_start"] = _first_match(text, r"\(EMI\)\s+(?:INR|Rs\.?)\s*[0-9,]+\s+\(EMI\)\s*([0-9/]+)")
    fields["interest_rate"] = _clean_value(_first_match(text, r"Interest rate \(%\).*?\s+([0-9.]+%\s+[A-Za-z]+(?:\s*\([^)]+\))?)"))
    fields["total_interest"] = _money(_first_match(text, r"Total Interest charged.*?Payable by the Borrower:\s*Rs\.?\s*([0-9,]+)"), "Rs.")
    fields["processing_fee"] = _money(_first_match(text, r"Processing fees.*?One time\s+Rs\.?\s*([0-9,]+)"), "Rs.")
    fields["insurance_fee"] = _money(_first_match(text, r"Insurance/Wellness charges.*?One time\s+Rs\.?\s*([0-9,]+)"), "Rs.")
    fields["valuation_fee"] = _money(_first_match(text, r"Valuation fees.*?One time\s+Rs\.?\s*([0-9,]+)"), "Rs.")
    fields["apr"] = _first_match(text, r"Annual Percentage Rate \(APR\) \(%\)\s*([0-9.]+\s*%)")
    fields["penal_charge"] = _clean_value(_first_match(text, r"delayed payment\s+([0-9.]+\s*%\s*p\.m\.(?:\s+is on outstanding\s+[A-Za-z\s]+?overdue)?)"))
    fields["foreclosure_charge"] = _first_match(text, r"Foreclosure charges[^A-Za-z0-9]+if applicable\s+([A-Za-z]+)")
    fields["part_payment_charge"] = _first_match(text, r"Part-payment Charges\s+([A-Za-z]+)")
    fields["validity"] = _first_match(text, r"valid for a period of ([A-Za-z0-9\s-]+?)(?:\s+\(|\s+Part 2|$)")
    return {key: value for key, value in fields.items() if value}


def _first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""
    return _clean_value(match.group(1))


def _clean_value(value: Optional[str]) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" .;,:")
    cleaned = re.sub(r"\s+%", "%", cleaned)
    cleaned = re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", cleaned)
    cleaned = re.sub(r"\bemi\b", "EMI", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bp\.m$", "p.m.", cleaned, flags=re.IGNORECASE)
    return cleaned


def _money(value: str, prefix: str) -> str:
    amount = _clean_value(value)
    if not amount:
        return ""
    return "{0} {1}".format(prefix, _format_amount(amount))


def _field_phrase(label: str, value: Optional[str]) -> str:
    if not value:
        return ""
    return "{0} {1}".format(label, value)


def _join_existing(values: Sequence[str]) -> str:
    return "; ".join(value for value in values if value)


def _repayment_phrases(fields: Dict[str, str]) -> List[str]:
    phrases = []
    if fields.get("installment_frequency"):
        phrases.append("{0} installments".format(fields["installment_frequency"]))
    if fields.get("epi_range"):
        phrases.append("{0} EPIs".format(fields["epi_range"]))
    if fields.get("emi"):
        phrases.append("EMI {0}".format(fields["emi"]))
    if fields.get("repayment_start"):
        phrases.append("repayment starts {0}".format(fields["repayment_start"]))
    return phrases


def _zero_fee_phrase(fields: Dict[str, str]) -> str:
    fee_items = []
    if fields.get("processing_fee"):
        fee_items.append("processing fee {0}".format(fields["processing_fee"]))
    if fields.get("insurance_fee"):
        fee_items.append("insurance/wellness fee {0}".format(fields["insurance_fee"]))
    if fields.get("valuation_fee"):
        fee_items.append("valuation fee {0}".format(fields["valuation_fee"]))
    if not fee_items:
        return ""
    return "fees: {0}".format(", ".join(fee_items))


def _format_amount(amount: str) -> str:
    if "," in amount:
        return amount
    if not re.fullmatch(r"\d+", amount):
        return amount
    return "{0:,}".format(int(amount))


def _sentence(text: str) -> str:
    cleaned = _clean_value(text)
    if cleaned and cleaned[-1] not in ".!?":
        return "{0}.".format(cleaned)
    return cleaned


def _split_units(text: str) -> List[str]:
    units = re.split(r"(?<=[.!?])\s+", text)
    split_units: List[str] = []
    for unit in units:
        split_units.extend(_HEADING_SPLIT_RE.split(unit))
    return split_units


def _clean_unit(unit: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(unit or "")).strip(" -;")
    cleaned = re.sub(r"^[,.:]+", "", cleaned).strip()
    return cleaned


def _is_substantive(candidate: str) -> bool:
    if not candidate:
        return False
    lower = candidate.lower()
    if lower.startswith(("citation:", "citations:", "source:", "sources:")):
        return False
    words = _WORD_RE.findall(candidate)
    if len(words) < 5:
        return False
    if candidate.endswith(":") and len(words) < 8:
        return False
    if _ends_in_incomplete_phrase(words):
        return False
    return True


def _trim_to_sentence(candidate: str, limit: int = MAX_SENTENCE_CHARS) -> str:
    cleaned = _clean_unit(candidate)
    if len(cleaned) > limit:
        window = cleaned[:limit].rstrip()
        boundary = max(window.rfind(". "), window.rfind("; "), window.rfind(", "))
        if boundary >= 80:
            cleaned = window[: boundary + 1].rstrip()
        else:
            cleaned = window.rsplit(" ", 1)[0].rstrip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = "{0}.".format(cleaned.rstrip(":;,"))
    return cleaned


def _dedupe_key(candidate: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", candidate.lower()).strip()


def _ends_in_incomplete_phrase(words: List[str]) -> bool:
    last = words[-1].strip(".,:;!?").lower() if words else ""
    return last in _INCOMPLETE_ENDINGS


def _is_near_duplicate(key: str, seen: set[str]) -> bool:
    key_words = set(key.split())
    if not key_words:
        return False
    for existing in seen:
        existing_words = set(existing.split())
        overlap = len(key_words.intersection(existing_words))
        if overlap >= 5 and overlap / float(max(len(key_words), len(existing_words))) > 0.82:
            return True
    return False
