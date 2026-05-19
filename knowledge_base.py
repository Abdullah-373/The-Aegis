"""Built-in knowledge base of contract risk patterns.

This is the corpus the RAG retrieval searches over. Each entry describes a
clause pattern that frequently appears in commercial contracts together with
the risk it implies and the standard mitigation. Specialist agents query
this corpus via the `search_precedent` tool before forming their analysis.

Keeping the corpus in code (rather than an external vector store) keeps the
project self-contained and removes the need for an embedding API call on
every query. A simple TF-IDF retriever is sufficient at this scale.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Precedent:
    id: str
    title: str
    pattern: str
    risk: str
    mitigation: str
    category: str

    def as_text(self) -> str:
        return (
            f"[{self.category}] {self.title}\n"
            f"Pattern: {self.pattern}\n"
            f"Risk: {self.risk}\n"
            f"Mitigation: {self.mitigation}"
        )


KNOWLEDGE_BASE: tuple[Precedent, ...] = (
    # -------------- LIABILITY --------------
    Precedent(
        id="liability_cap_short",
        title="Short liability cap (under 12 months fees)",
        pattern="liability cap limited to fees paid in the last 3, 6, or 12 months",
        risk="A single data breach or contractual failure can vastly exceed the cap, leaving the buyer effectively self-insured.",
        mitigation="Negotiate the cap to at least 12 months of annual fees. Carve out gross negligence, wilful misconduct, data breach, and IP indemnification from the cap.",
        category="liability",
    ),
    Precedent(
        id="liability_cap_zero_carveouts",
        title="Liability cap without carve-outs for IP or data",
        pattern="single aggregate liability cap with no exceptions for confidentiality, IP infringement, or data protection",
        risk="The most severe categories of harm (data leaks, IP infringement) are capped at a low number.",
        mitigation="Add supercaps for data and IP claims, typically 2-3x the general cap, or remove the cap for those categories entirely.",
        category="liability",
    ),
    Precedent(
        id="consequential_damages_waiver",
        title="Mutual waiver of consequential damages",
        pattern="parties waive all indirect, incidental, consequential, special, or punitive damages",
        risk="Often acceptable but blocks recovery of lost profits and reputational harm — disproportionate when the vendor is the sole source of failure.",
        mitigation="Carve out the vendor's breach of confidentiality, willful misconduct, and IP infringement from the waiver.",
        category="liability",
    ),
    # -------------- INDEMNIFICATION --------------
    Precedent(
        id="one_way_indemnification",
        title="One-way indemnification (customer indemnifies vendor only)",
        pattern="customer agrees to defend and indemnify vendor, but no reciprocal vendor indemnification",
        risk="The customer bears all third-party legal risk while the vendor bears none. Highly asymmetric.",
        mitigation="Insist on reciprocal indemnification at least for IP infringement claims arising from the vendor's product.",
        category="indemnification",
    ),
    Precedent(
        id="ip_indemnification_missing",
        title="No IP indemnification from vendor",
        pattern="silence on what happens if vendor's product infringes a third party's IP rights",
        risk="Customer can be sued by a third party for use of the vendor's product with no recourse.",
        mitigation="Require vendor IP indemnification with defence, settlement authority, and a 'modify or replace' obligation.",
        category="indemnification",
    ),
    # -------------- TERMINATION --------------
    Precedent(
        id="termination_for_convenience_penalty",
        title="Heavy early-termination penalty (75-100% of contract value)",
        pattern="customer must pay a large percentage of the remaining contract value to terminate for convenience",
        risk="Customer is locked into a bad deal with no economic exit. May be unenforceable as a penalty in some jurisdictions.",
        mitigation="Cap convenience-termination fees at 50% or below, or limit to demonstrable wasted costs. Verify enforceability under the governing law.",
        category="termination",
    ),
    Precedent(
        id="auto_renewal_long_notice",
        title="Auto-renewal with long opt-out notice period",
        pattern="contract auto-renews unless 90+ days notice is given before the end of the term",
        risk="A missed notice window locks the customer in for another full term, often at a higher rate.",
        mitigation="Reduce the notice period to 30 days, or require the vendor to provide explicit renewal notice before the window opens.",
        category="termination",
    ),
    Precedent(
        id="cure_period_long",
        title="Long cure period before customer can terminate for material breach",
        pattern="vendor has 60 or 90 days to cure a material breach before the customer can terminate",
        risk="A broken service can persist for months with no termination right, while fees continue to accrue.",
        mitigation="Reduce cure period to 30 days for non-payment and 14 days for SLA breaches. Allow immediate termination for repeat breaches.",
        category="termination",
    ),
    # -------------- FEES & PRICING --------------
    Precedent(
        id="annual_fee_uplift_unilateral",
        title="Unilateral annual fee uplift",
        pattern="vendor may raise fees by a fixed percentage (e.g. 10-15%) at each renewal without negotiation",
        risk="Compounded over a multi-year contract this can double or triple the effective price. The customer has no leverage to push back.",
        mitigation="Cap the uplift at CPI + 2%, or require a benchmarking clause. Reset pricing every 2 years.",
        category="pricing",
    ),
    Precedent(
        id="late_payment_high_interest",
        title="High late-payment interest rate",
        pattern="late payments accrue interest at 1.5% per month or higher",
        risk="18% annualised is well above commercial rates and can be unenforceable as a penalty in some jurisdictions.",
        mitigation="Reduce to a commercially reasonable rate (e.g. base rate + 2%).",
        category="pricing",
    ),
    Precedent(
        id="non_refundable_prepayment",
        title="Fully non-refundable annual prepayment",
        pattern="annual fees paid in advance and explicitly non-refundable under any circumstances",
        risk="If the vendor fails to deliver, the customer cannot recover unused fees.",
        mitigation="Make fees pro-rata refundable on termination for cause. Tie refunds to SLA credits.",
        category="pricing",
    ),
    # -------------- DATA & PRIVACY --------------
    Precedent(
        id="perpetual_data_license",
        title="Perpetual licence over anonymised customer data",
        pattern="customer grants vendor a perpetual, irrevocable, royalty-free licence over anonymised or aggregated customer data",
        risk="Vendor can use customer data to train ML models or build derivative products that compete with the customer. 'Anonymisation' is rarely irreversible.",
        mitigation="Restrict the licence to the term of the agreement. Forbid use for training models that will be sold to third parties. Require deletion on termination.",
        category="data",
    ),
    Precedent(
        id="data_residency_silent",
        title="Silence on data residency",
        pattern="no clause specifying where customer data is stored or processed",
        risk="Data may flow to jurisdictions that conflict with the customer's regulatory obligations (GDPR, HIPAA, regional residency laws).",
        mitigation="Add a clause restricting storage and processing to named jurisdictions. Require notice and consent for any change.",
        category="data",
    ),
    Precedent(
        id="breach_notification_long",
        title="Slow breach-notification timeline",
        pattern="vendor must notify customer of a data breach within 30 or more days",
        risk="GDPR requires notification within 72 hours. A slow vendor notification can put the customer in regulatory non-compliance.",
        mitigation="Require breach notification within 72 hours of discovery, with preliminary notice within 24 hours.",
        category="data",
    ),
    Precedent(
        id="sub_processor_consent_implicit",
        title="Sub-processor changes without explicit consent",
        pattern="vendor may add or change sub-processors with notice only, no consent required",
        risk="Customer's data can end up in the hands of new third parties without an opportunity to object.",
        mitigation="Require prior written consent for new sub-processors, or at least a right to object and terminate.",
        category="data",
    ),
    # -------------- IP --------------
    Precedent(
        id="customer_ip_assignment",
        title="Assignment of customer-developed IP to vendor",
        pattern="any improvements, derivative works, or feedback created using the platform are owned by the vendor",
        risk="Customer's own contributions become vendor property — particularly damaging in joint development scenarios.",
        mitigation="Limit assignment to feedback only. Customer retains ownership of all derivative works of its own data and processes.",
        category="ip",
    ),
    Precedent(
        id="feedback_broad_license",
        title="Broad feedback licence",
        pattern="customer grants vendor a perpetual, worldwide licence over all feedback, suggestions, and ideas",
        risk="Common and usually acceptable, but combined with other clauses can transfer significant IP value.",
        mitigation="Limit to non-confidential feedback. Disclaim any obligation to share trade-secret-level ideas.",
        category="ip",
    ),
    # -------------- DISPUTES --------------
    Precedent(
        id="forced_arbitration_distant",
        title="Forced arbitration in a distant jurisdiction",
        pattern="all disputes settled by binding arbitration in a jurisdiction far from the customer's domicile",
        risk="High cost of pursuing claims may deter the customer from enforcing rights. Vendor home-court advantage.",
        mitigation="Allow arbitration in either party's home jurisdiction. Reserve injunctive-relief claims for court.",
        category="disputes",
    ),
    Precedent(
        id="jury_trial_waiver",
        title="Jury trial waiver",
        pattern="parties waive any right to a jury trial",
        risk="In some jurisdictions (US) this removes a significant procedural advantage. Less impactful in civil-law jurisdictions.",
        mitigation="Strike for US-based customers. Acceptable in jurisdictions where jury trials are uncommon.",
        category="disputes",
    ),
    Precedent(
        id="class_action_waiver",
        title="Class-action waiver",
        pattern="parties waive any right to bring or participate in a class action",
        risk="Limits collective bargaining power on systemic issues that affect many customers similarly.",
        mitigation="Acceptable for enterprise B2B; resist strongly for consumer or small-business contracts.",
        category="disputes",
    ),
    # -------------- SLA --------------
    Precedent(
        id="sla_vague_commitment",
        title="Vague SLA commitment ('commercially reasonable efforts')",
        pattern="vendor commits to a percentage uptime using 'commercially reasonable efforts'",
        risk="The standard is subjective and almost impossible to enforce. No specific remedy is triggered.",
        mitigation="Replace with hard percentage commitments measured monthly, with automatic service credits or termination rights below a floor.",
        category="sla",
    ),
    Precedent(
        id="sla_no_credits",
        title="SLA without service credits",
        pattern="uptime commitment exists but there is no remedy when the vendor misses it",
        risk="The SLA is decorative — the customer has no leverage when service degrades.",
        mitigation="Require automatic service credits (typically 5-25% of monthly fees) for each downtime tier, and termination for chronic underperformance.",
        category="sla",
    ),
    Precedent(
        id="sla_exclusions_broad",
        title="Broad SLA exclusions (scheduled maintenance, third-party failure)",
        pattern="SLA excludes scheduled maintenance, force majeure, third-party failures, and any cause outside vendor control",
        risk="With wide exclusions the effective uptime can be far below the stated number.",
        mitigation="Cap scheduled-maintenance exclusions to a low monthly maximum (e.g. 4 hours). Require notice for maintenance.",
        category="sla",
    ),
    # -------------- ASSIGNMENT --------------
    Precedent(
        id="assignment_asymmetric",
        title="Asymmetric assignment rights",
        pattern="vendor may freely assign the contract; customer needs vendor's prior written consent",
        risk="Vendor can be acquired by a competitor or undesirable party and the customer has no exit. Customer is stuck.",
        mitigation="Make assignment rights symmetric, or add a customer termination right on vendor change of control.",
        category="assignment",
    ),
    Precedent(
        id="change_of_control_silent",
        title="Silence on change of control",
        pattern="no clause addressing what happens if either party changes ownership",
        risk="The contract continues unchanged under new ownership — may be undesirable if the vendor is acquired by a competitor.",
        mitigation="Add a termination right for either party on the other's change of control.",
        category="assignment",
    ),
    # -------------- CONFIDENTIALITY --------------
    Precedent(
        id="confidentiality_short_survival",
        title="Short confidentiality survival",
        pattern="confidentiality obligations survive only 1-2 years after termination",
        risk="Trade secrets typically need protection indefinitely. A short survival window exposes them.",
        mitigation="Trade secrets survive indefinitely. Other confidential information survives 5+ years.",
        category="confidentiality",
    ),
    Precedent(
        id="confidentiality_no_residuals",
        title="No residual-knowledge carve-out",
        pattern="receiving party may not use any information learned during the engagement, even from memory",
        risk="Practically impossible to enforce; can hobble normal employee mobility.",
        mitigation="Add a residual-knowledge clause permitting general (non-specific) skills and learnings to be retained.",
        category="confidentiality",
    ),
    # -------------- GOVERNING LAW --------------
    Precedent(
        id="governing_law_unfavourable",
        title="Governing law in a vendor-friendly jurisdiction",
        pattern="agreement governed by laws of a jurisdiction chosen for vendor convenience (Delaware, Cayman, etc.)",
        risk="The customer must litigate under unfamiliar law with non-domestic counsel.",
        mitigation="Negotiate to the customer's home jurisdiction or a neutral common-law jurisdiction (UK, Singapore).",
        category="governing_law",
    ),
    # -------------- WARRANTY --------------
    Precedent(
        id="as_is_warranty_disclaimer",
        title="'As-is' warranty disclaimer",
        pattern="vendor disclaims all warranties, express or implied, including merchantability and fitness for purpose",
        risk="Customer has no contractual quality recourse beyond explicit SLAs.",
        mitigation="Require an express warranty of conformance to documentation. Disclaimers acceptable for free trial or beta features only.",
        category="warranty",
    ),
    # -------------- EXIT --------------
    Precedent(
        id="data_export_unclear",
        title="No clear data-export mechanism on termination",
        pattern="agreement is silent on how the customer recovers its data when the contract ends",
        risk="Customer faces lock-in. May lose data entirely or face hefty 'transition fees' from vendor.",
        mitigation="Specify a defined export format (CSV/JSON/parquet), a deadline, and a fixed (or zero) transition fee.",
        category="exit",
    ),
    Precedent(
        id="transition_assistance_paid",
        title="Paid-only transition assistance",
        pattern="vendor provides post-termination data export only at additional cost",
        risk="Combined with non-refundable prepayment, this can multiply the effective exit cost.",
        mitigation="Require 60-90 days of free export assistance at termination, regardless of reason.",
        category="exit",
    ),
    # -------------- COMPLIANCE --------------
    Precedent(
        id="compliance_silent_on_certifications",
        title="No commitment to maintain security certifications",
        pattern="vendor describes current certifications (SOC 2, ISO 27001) but does not promise to maintain them",
        risk="Customer compliance posture depends on vendor certs that may lapse silently.",
        mitigation="Require ongoing maintenance of specified certifications with annual attestation.",
        category="compliance",
    ),
    Precedent(
        id="subcontractor_compliance_loose",
        title="Loose sub-contractor compliance flow-down",
        pattern="vendor agrees to ensure sub-contractor compliance with reasonable efforts",
        risk="Sub-contractors are often where breaches actually occur; reasonable-efforts standards are weak.",
        mitigation="Require flow-down of all data-protection and security obligations on the same terms.",
        category="compliance",
    ),
    # -------------- AUDIT --------------
    Precedent(
        id="audit_rights_missing",
        title="No customer audit rights",
        pattern="customer has no right to audit vendor security, compliance, or financial records",
        risk="Customer must take vendor compliance claims on trust, with no verification mechanism.",
        mitigation="Add an annual audit right (third-party or self) with reasonable notice and confidentiality protections.",
        category="audit",
    ),
)


# ---------------------------------------------------------------------------
# Retrieval (TF-IDF + cosine similarity, pure Python)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "a", "an", "and", "or", "the", "of", "to", "in", "for", "by",
    "is", "are", "with", "on", "at", "be", "as", "it", "this", "that",
    "from", "any", "all", "no", "not",
})


def _tokenize(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _term_freq(tokens: Iterable[str]) -> Counter[str]:
    return Counter(tokens)


def _build_idf(docs: list[list[str]]) -> dict[str, float]:
    n_docs = len(docs)
    df: Counter[str] = Counter()
    for d in docs:
        for term in set(d):
            df[term] += 1
    return {
        term: math.log((n_docs + 1) / (count + 1)) + 1.0
        for term, count in df.items()
    }


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = _term_freq(tokens)
    if not tf:
        return {}
    max_tf = max(tf.values())
    return {
        term: (count / max_tf) * idf.get(term, 0.0)
        for term, count in tf.items()
    }


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    num = sum(a[t] * b[t] for t in shared)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


# Precompute the corpus once on import.
_CORPUS_DOCS: list[list[str]] = [_tokenize(p.as_text()) for p in KNOWLEDGE_BASE]
_IDF: dict[str, float] = _build_idf(_CORPUS_DOCS)
_CORPUS_VECTORS: list[dict[str, float]] = [_tfidf_vector(d, _IDF) for d in _CORPUS_DOCS]


def search(query: str, *, k: int = 4) -> list[Precedent]:
    """Return the top-k precedents most similar to the query."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    q_vec = _tfidf_vector(q_tokens, _IDF)
    scored = [
        (i, _cosine(q_vec, _CORPUS_VECTORS[i]))
        for i in range(len(KNOWLEDGE_BASE))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [KNOWLEDGE_BASE[i] for i, score in scored[:k] if score > 0]


def categories() -> list[str]:
    return sorted({p.category for p in KNOWLEDGE_BASE})
