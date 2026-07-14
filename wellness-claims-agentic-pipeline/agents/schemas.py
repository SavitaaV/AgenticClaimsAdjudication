"""
schemas.py
──────────
Data contracts for the WSA Claims Adjudication Pipeline.

Three agents, three schemas:
  ExtractorOutput   ← Haiku 4.5  reads the receipt, extracts fields only
  ValidatorOutput   ← Sonnet 4.6 runs three-gate test + balance check, no decision
  AdjudicatorOutput ← Opus 4.8   reasons over validator output, recommends + scores

Rule: agents communicate via JSON only.
      Each schema has to_dict() and from_dict() for serialisation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# ═══════════════════════════════════════════════════════════════════════
# PLAN CONSTANTS
# Source of truth for the plan configuration.
# Change here → changes everywhere.
# ═══════════════════════════════════════════════════════════════════════

PLAN = {
    "name":              "Maple Shield WSA",
    "benefit_year":      "2025",
    "year_start":        "2025-01-01",
    "year_end":          "2025-12-31",
    "runout_deadline":   "2026-03-31",   # 90 days after year end
    "single_allocation": 750.00,
    "family_allocation": 1000.00,
    "reimbursement_pct": 1.0,            # 100%
    "authority":         "Where the booklet and manual differ, the booklet governs.",
}

# Reason codes from the adjudication manual
REASON_CODES = {
    "A-OK":    "Approved in full",
    "A-PART":  "Approved partial — balance insufficient for full amount",
    "D-PERS":  "Denied — claimant is not an eligible person",
    "D-EXP":   "Denied — expense is not an eligible WSA category",
    "D-PER":   "Denied — expense was incurred outside the benefit period",
    "D-DOC":   "Denied — documentation is invalid or insufficient",
    "D-BAL":   "Denied — available balance is zero",
    "P-DOC":   "Pending — documentation required before decision can be made",
    "P-REV":   "Pending — claim is under review (duplicate or fraud indicator)",
    "R-HCSA":  "Routed — expense is a CRA medical expense; submit to HCSA instead",
}

# WSA eligible categories (from manual Section 9)
WSA_CATEGORIES = {
    "WSA-FIT": "Fitness — gym, sports, equipment, footwear, classes, ski passes",
    "WSA-NUT": "Nutrition — dietitian, vitamins, weight-management programs",
    "WSA-MEN": "Mental wellness — meditation apps, counselling, life coaching",
    "WSA-FAM": "Family — childcare, elder care, pet care, domestic services",
    "WSA-DEV": "Development — tuition, courses, books, professional memberships",
    "WSA-SAF": "Safety & ergonomics — ergonomic furniture, home safety equipment",
    "WSA-ALT": "Alternative wellness — massage, acupuncture, naturopathy (only when not a CRA medical expense AND not covered by the health plan)",
    "WSA-FIN": "Financial & professional services — RRSP/RESP/TFSA contributions, estate planning, legal fees, certain insurance premiums",
}


# ═══════════════════════════════════════════════════════════════════════
# SCHEMA 1 — EXTRACTOR OUTPUT
# What Haiku 4.5 returns after reading the receipt.
# Job: extract only. No rules. No judgment. No decision.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExtractorOutput:
    # Core fields pulled directly from the receipt
    vendor:           Optional[str]   = None  # Business name on receipt
    date_incurred:    Optional[str]   = None  # YYYY-MM-DD service/purchase date
    amount_pretax:    Optional[float] = None  # Subtotal before tax
    amount_total:     Optional[float] = None  # Total amount paid including tax
    tax_amount:       Optional[float] = None  # Tax charged (HST/GST)
    items:            list            = field(default_factory=list)  # Line items
    claimant_name:    Optional[str]   = None  # Name on receipt
    payment_confirmed: bool           = False  # True if receipt shows PAID/APPROVED
    payment_method:   Optional[str]   = None  # Visa, Mastercard, Debit, etc.

    # Quality signals — used by validator and adjudicator
    unreadable_fields: list = field(default_factory=list)  # Fields that could not be read
    image_quality:    str   = "good"   # "good" | "poor" | "unreadable"
    document_type:    str   = "receipt"  # "receipt" | "invoice" | "quote" | "email" | "unknown"

    # Extractor's own notes — what it observed, questions it has
    extraction_notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractorOutput":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> "ExtractorOutput":
        return cls.from_dict(json.loads(s))


# ═══════════════════════════════════════════════════════════════════════
# SCHEMA 2 — VALIDATOR OUTPUT
# What Sonnet 4.6 returns after running the three-gate test.
# Job: verify facts against the rules. Surface findings. NO final decision.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GateResult:
    """Result for a single gate in the three-gate test."""
    passed:   bool
    finding:  str            # Plain English finding — what was checked and what was found
    citation: Optional[str] = None  # Manual section or booklet page that applies

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BalanceCheck:
    """Balance availability check — runs after all three gates pass."""
    available_balance: float
    claim_amount:      float
    sufficient:        bool
    partial_amount:    Optional[float] = None  # Amount approvable if balance < claim

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidatorOutput:
    # ── Three gates (assessed in order) ───────────────────────────────
    gate1_eligible_person:  Optional[GateResult] = None
    gate2_eligible_expense: Optional[GateResult] = None
    gate2_in_period:        Optional[GateResult] = None  # Period check is part of gate 2
    gate3_documentation:    Optional[GateResult] = None

    # ── Balance check (only reached if all 3 gates pass) ──────────────
    balance_check: Optional[BalanceCheck] = None

    # ── Special routing and flags ──────────────────────────────────────
    is_medical_expense:  bool           = False  # True → should go to HCSA
    route_to_hcsa:       bool           = False
    category_code:       Optional[str]  = None   # e.g. WSA-FIT, WSA-MEN
    duplicate_flag:      bool           = False  # Matches another claim in this batch
    duplicate_of:        Optional[str]  = None   # Claim ID it duplicates

    # ── Validator's structured summary ────────────────────────────────
    # NOT a decision — these are findings for the adjudicator to reason over
    overall_gate_status:  str = "INCOMPLETE"  # PASS | FAIL | PARTIAL | PENDING | HCSA
    first_failing_gate:   Optional[str] = None  # Which gate failed first
    validator_notes:      str = ""             # Anything noteworthy for the adjudicator

    def to_dict(self) -> dict:
        d = {
            "gate1_eligible_person":  self.gate1_eligible_person.to_dict()  if self.gate1_eligible_person  else None,
            "gate2_eligible_expense": self.gate2_eligible_expense.to_dict() if self.gate2_eligible_expense else None,
            "gate2_in_period":        self.gate2_in_period.to_dict()        if self.gate2_in_period        else None,
            "gate3_documentation":    self.gate3_documentation.to_dict()    if self.gate3_documentation    else None,
            "balance_check":          self.balance_check.to_dict()          if self.balance_check          else None,
            "is_medical_expense":     self.is_medical_expense,
            "route_to_hcsa":          self.route_to_hcsa,
            "category_code":          self.category_code,
            "duplicate_flag":         self.duplicate_flag,
            "duplicate_of":           self.duplicate_of,
            "overall_gate_status":    self.overall_gate_status,
            "first_failing_gate":     self.first_failing_gate,
            "validator_notes":        self.validator_notes,
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ValidatorOutput":
        obj = cls()
        if d.get("gate1_eligible_person"):
            obj.gate1_eligible_person = GateResult(**d["gate1_eligible_person"])
        if d.get("gate2_eligible_expense"):
            obj.gate2_eligible_expense = GateResult(**d["gate2_eligible_expense"])
        if d.get("gate2_in_period"):
            obj.gate2_in_period = GateResult(**d["gate2_in_period"])
        if d.get("gate3_documentation"):
            obj.gate3_documentation = GateResult(**d["gate3_documentation"])
        if d.get("balance_check"):
            obj.balance_check = BalanceCheck(**d["balance_check"])
        for key in ["is_medical_expense", "route_to_hcsa", "category_code",
                    "duplicate_flag", "duplicate_of", "overall_gate_status",
                    "first_failing_gate", "validator_notes"]:
            if key in d:
                setattr(obj, key, d[key])
        return obj

    @classmethod
    def from_json(cls, s: str) -> "ValidatorOutput":
        return cls.from_dict(json.loads(s))


# ═══════════════════════════════════════════════════════════════════════
# SCHEMA 3 — ADJUDICATOR OUTPUT
# What Opus 4.8 returns after reasoning over extractor + validator outputs.
# Job: synthesise, recommend, score, explain. Collaborative tone.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AdjudicatorOutput:
    # ── The recommendation ────────────────────────────────────────────
    # "APPROVE" | "DENY" | "PEND" | "ROUTE_HCSA" | "ESCALATE"
    decision:      str = "ESCALATE"
    reason_code:   str = "P-REV"       # From REASON_CODES above

    # ── Amounts ───────────────────────────────────────────────────────
    approved_amount: float = 0.0
    denied_amount:   float = 0.0

    # ── Confidence score ──────────────────────────────────────────────
    # Range: 0.0 (no confidence) → 1.0 (fully confident)
    # Must include HOW the score was calculated (see confidence_reasoning)
    confidence_score:     float = 0.0
    confidence_reasoning: str   = ""   # Explicit breakdown of how score was derived

    # ── Explanation (plain English, examiner-facing) ──────────────────
    rationale:    str  = ""   # 2-4 sentences explaining the recommendation
    cited_rules:  list = field(default_factory=list)  # ["Manual §X.X — ...", "Booklet p.X — ..."]

    # ── Payroll tax flag (required on every approval per manual §4.2) ─
    tax_flag: bool          = False
    tax_note: Optional[str] = None

    # ── Examiner guidance ─────────────────────────────────────────────
    examiner_suggestions: str          = ""    # What the examiner should specifically look at
    uncertainty_notes:    Optional[str] = None  # What the agent is uncertain about
    i_dont_know:          bool          = False  # True if agent genuinely cannot determine

    # ── Feedback and Q&A ─────────────────────────────────────────────
    # Populated when examiner asks a follow-up question
    qa_answer: Optional[str] = None
    # Notes if prior feedback was incorporated into this recommendation
    feedback_incorporated: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "AdjudicatorOutput":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> "AdjudicatorOutput":
        return cls.from_dict(json.loads(s))


# ═══════════════════════════════════════════════════════════════════════
# COMPLETE CLAIM RECORD
# The full record for one claim — wraps all three agent outputs
# plus examiner decision tracking and feedback.
# Stored in claims_db: { claim_id: ClaimRecord }
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ClaimRecord:
    # ── Identity ──────────────────────────────────────────────────────
    claim_id:       str = ""
    file_name:      str = ""
    file_path:      str = ""
    claimant_name:  str = ""
    coverage_type:  str = "single"   # "single" | "family"
    submitted_at:   str = ""

    # ── Agent outputs (populated as pipeline runs) ────────────────────
    extractor_output:   Optional[ExtractorOutput]   = None
    validator_output:   Optional[ValidatorOutput]   = None
    adjudicator_output: Optional[AdjudicatorOutput] = None

    # ── Current status ────────────────────────────────────────────────
    # "pending_ai" → "awaiting_review" → "approved"/"rejected"/"pended"/"escalated"/"routed"
    status: str = "pending_ai"

    # ── Triage tag (set by pipeline after adjudicator runs) ──────────
    # "EASY"   → high confidence, clean claim, straightforward decision
    # "REVIEW" → low confidence, ambiguous, or flagged — needs examiner attention
    triage_tag: str = "REVIEW"

    # ── Examiner decision (set when human acts) ───────────────────────
    examiner_decision:  Optional[str] = None   # "approved" | "rejected" | "pended" | "escalated" | "routed"
    examiner_notes:     str           = ""
    override_reason:    str           = ""     # Required if overriding agent recommendation
    feedback_note:      str           = ""     # Free text — fed back into adjudicator over time
    decided_at:         Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "claim_id":          self.claim_id,
            "file_name":         self.file_name,
            "file_path":         self.file_path,
            "claimant_name":     self.claimant_name,
            "coverage_type":     self.coverage_type,
            "submitted_at":      self.submitted_at,
            "status":            self.status,
            "triage_tag":        self.triage_tag,
            "examiner_decision": self.examiner_decision,
            "examiner_notes":    self.examiner_notes,
            "override_reason":   self.override_reason,
            "feedback_note":     self.feedback_note,
            "decided_at":        self.decided_at,
            "extractor_output":   self.extractor_output.to_dict()   if self.extractor_output   else None,
            "validator_output":   self.validator_output.to_dict()   if self.validator_output   else None,
            "adjudicator_output": self.adjudicator_output.to_dict() if self.adjudicator_output else None,
        }


# ═══════════════════════════════════════════════════════════════════════
# QUICK TEST — run this file directly to verify schemas work
# python schemas.py
# Expected output: ✓ All schemas instantiate and serialise correctly
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing schemas...\n")

    # Test ExtractorOutput
    ex = ExtractorOutput(
        vendor="GoodLife Fitness",
        date_incurred="2025-03-14",
        amount_pretax=150.00,
        amount_total=169.50,
        tax_amount=19.50,
        items=["Membership - 6 month"],
        claimant_name="Jordan Tran",
        payment_confirmed=True,
        payment_method="Visa ****4127",
        image_quality="good",
        document_type="receipt",
        extraction_notes="All fields clearly readable."
    )
    assert ex.to_dict()["vendor"] == "GoodLife Fitness"
    assert ExtractorOutput.from_dict(ex.to_dict()).vendor == "GoodLife Fitness"
    print("  ✓ ExtractorOutput")

    # Test ValidatorOutput
    val = ValidatorOutput(
        gate1_eligible_person  = GateResult(passed=True,  finding="Claimant Jordan Tran is eligible.", citation="Booklet — Who Is Covered"),
        gate2_eligible_expense = GateResult(passed=True,  finding="Gym membership — WSA-FIT.", citation="Manual §9.1"),
        gate2_in_period        = GateResult(passed=True,  finding="Date 2025-03-14 is within 2025 benefit year.", citation="Manual §5.2"),
        gate3_documentation    = GateResult(passed=True,  finding="Receipt shows all required fields. Paid in full.", citation="Manual §8.1"),
        balance_check          = BalanceCheck(available_balance=750.00, claim_amount=169.50, sufficient=True),
        category_code          = "WSA-FIT",
        overall_gate_status    = "PASS",
        validator_notes        = "Clean claim. No flags."
    )
    assert val.to_dict()["category_code"] == "WSA-FIT"
    assert ValidatorOutput.from_dict(val.to_dict()).category_code == "WSA-FIT"
    print("  ✓ ValidatorOutput")

    # Test AdjudicatorOutput
    adj = AdjudicatorOutput(
        decision              = "APPROVE",
        reason_code           = "A-OK",
        approved_amount       = 169.50,
        denied_amount         = 0.0,
        confidence_score      = 0.95,
        confidence_reasoning  = "All gates passed. Receipt clear. Category unambiguous. Minor uncertainty: claimant eligibility not independently verified (0.05 deduction).",
        rationale             = "This looks like a clean, eligible claim. GoodLife Fitness is a recognised fitness facility, and a 6-month gym membership falls squarely under WSA-FIT. The receipt is clear, payment is confirmed, and the date is within the benefit year. I'd suggest approving this one.",
        cited_rules           = ["Manual §9.1 — Fitness facilities eligible under WSA-FIT", "Booklet — How much you get back (100%)"],
        tax_flag              = True,
        tax_note              = "WSA reimbursement is a taxable benefit. Flag for payroll source deductions.",
        examiner_suggestions  = "Nothing unusual here. Standard approval.",
    )
    assert adj.to_dict()["decision"] == "APPROVE"
    assert AdjudicatorOutput.from_dict(adj.to_dict()).reason_code == "A-OK"
    print("  ✓ AdjudicatorOutput")

    # Test ClaimRecord
    claim = ClaimRecord(
        claim_id       = "007",
        file_name      = "07_gym_membership.jpg",
        claimant_name  = "Jordan Tran",
        coverage_type  = "single",
        submitted_at   = "2025-06-28T10:00:00",
        status         = "awaiting_review",
        triage_tag     = "EASY",
        extractor_output   = ex,
        validator_output   = val,
        adjudicator_output = adj,
    )
    d = claim.to_dict()
    assert d["claim_id"] == "007"
    assert d["extractor_output"]["vendor"] == "GoodLife Fitness"
    assert d["adjudicator_output"]["confidence_score"] == 0.95
    print("  ✓ ClaimRecord\n")

    print("✓ All schemas instantiate and serialise correctly")
    print("✓ Phase 1 complete — proceed to Phase 2 (system prompts)")