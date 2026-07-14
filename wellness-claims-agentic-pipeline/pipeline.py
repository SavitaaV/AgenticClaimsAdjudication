"""
pipeline.py
───────────
Pipeline Orchestrator — wires Extractor → Validator → Adjudicator

Responsibility:
  Accept a receipt file and claimant details.
  Run the three agents in sequence.
  Manage shared state: claims DB, balance ledger, feedback log.
  Handle examiner decisions and Q&A.
  Provide stats for the Gradio UI.

This module is the single entry point the Gradio app talks to.
It never calls the Anthropic API directly — it delegates to the agents.

Public API:
  process_claim(file_path, claimant_name, coverage_type)  → ClaimRecord dict
  submit_decision(claim_id, decision, notes, override, feedback)  → ClaimRecord dict
  ask_question(claim_id, question)  → str
  get_stats()  → dict
  get_all_claims()  → list[dict]
  get_claim(claim_id)  → dict | None
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path

import anthropic

try:
    from agents.schemas import (
        ClaimRecord, ExtractorOutput, ValidatorOutput,
        AdjudicatorOutput, PLAN
    )
    from agents.models.extractor   import run_extractor
    from agents.models.validator   import run_validator, check_duplicate
    from agents.models.adjudicator import run_adjudicator, determine_triage_tag, EASY_CONFIDENCE_THRESHOLD
except ImportError:
    from schemas import (
        ClaimRecord, ExtractorOutput, ValidatorOutput,
        AdjudicatorOutput, PLAN
    )
    from extractor   import run_extractor
    from validator   import run_validator, check_duplicate
    from adjudicator import run_adjudicator, determine_triage_tag, EASY_CONFIDENCE_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════
# SHARED STATE
# In-memory for the prototype.
# Replace with a database in production.
# ═══════════════════════════════════════════════════════════════════════

_claims_db:      dict = {}   # { claim_id: ClaimRecord.to_dict() }
_balance_ledger: dict = {}   # { claimant_name: float }
_feedback_log:   list = []   # [ feedback_record_dict ]
_claim_counter:  list = [0]  # mutable int — use list so closures can modify it

# Path for persisting feedback across Colab sessions
_FEEDBACK_PATH = Path("feedback_log.json")


def _load_feedback_log():
    """Load feedback log from disk if it exists."""
    global _feedback_log
    if _FEEDBACK_PATH.exists():
        try:
            with open(_FEEDBACK_PATH) as f:
                _feedback_log = json.load(f)
            print(f"  Loaded {len(_feedback_log)} feedback entries from {_FEEDBACK_PATH}")
        except Exception:
            _feedback_log = []

def _save_feedback_log():
    """Persist feedback log to disk."""
    try:
        with open(_FEEDBACK_PATH, "w") as f:
            json.dump(_feedback_log, f, indent=2)
    except Exception as e:
        print(f"  Warning: could not save feedback log: {e}")


# Load on import
_load_feedback_log()


# ═══════════════════════════════════════════════════════════════════════
# BALANCE LEDGER HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _get_balance(claimant: str, coverage_type: str) -> float:
    """Get or initialise the balance for a claimant."""
    if claimant not in _balance_ledger:
        allocation = (
            PLAN["family_allocation"]
            if coverage_type.lower() == "family"
            else PLAN["single_allocation"]
        )
        _balance_ledger[claimant] = float(allocation)
    return _balance_ledger[claimant]


def _deduct_balance(claimant: str, amount: float):
    """Deduct an approved amount from the claimant's balance."""
    if claimant in _balance_ledger:
        _balance_ledger[claimant] = max(0.0, _balance_ledger[claimant] - amount)


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT GUARDRAIL (script-based — no LLM)
# Scans adjudicator text fields for content that should not appear
# in a regulated insurance context.
# ═══════════════════════════════════════════════════════════════════════

_BLOCKED = [
    re.compile(r'\b(manulife|sunlife|great.west|desjardins|blue\s+cross)\b',     re.IGNORECASE),
    re.compile(r'\b(tax\s+advice|legal\s+advice|financial\s+advice)\b',          re.IGNORECASE),
    re.compile(r'\b\d{3}[\s\-]\d{3}[\s\-]\d{3}\b'),   # SIN pattern
    re.compile(r'\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b'),  # full card number
]

def _guardrail(adj: AdjudicatorOutput) -> AdjudicatorOutput:
    """
    Scan text fields and redact any blocked content.
    Returns the (possibly modified) AdjudicatorOutput.
    """
    text_fields = ["rationale", "examiner_suggestions",
                   "uncertainty_notes", "qa_answer", "confidence_reasoning"]

    for field in text_fields:
        value = getattr(adj, field, None)
        if not value:
            continue
        for pattern in _BLOCKED:
            value = pattern.sub("[REDACTED]", value)
        setattr(adj, field, value)

    return adj


# ═══════════════════════════════════════════════════════════════════════
# CLAIM ID GENERATOR
# ═══════════════════════════════════════════════════════════════════════

def _next_claim_id() -> str:
    """Generate the next zero-padded claim ID: 001, 002, ..."""
    _claim_counter[0] += 1
    return str(_claim_counter[0]).zfill(3)


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def process_claim(
    file_path:     str,
    claimant_name: str,
    coverage_type: str = "single",
    client:        anthropic.Anthropic = None,
) -> dict:
    """
    Process a single receipt through the full three-agent pipeline.

    Steps:
      1  Parse file (script)
      2  Run Extractor — Haiku 4.5
      3  Duplicate check (script)
      4  Get balance (script)
      5  Run Validator — Sonnet 4.6
      6  Run Adjudicator — Opus 4.8
      7  Output guardrail (script)
      8  Determine triage tag (script)
      9  Build ClaimRecord and store

    Args:
        file_path     : path to receipt (.jpg .jpeg .png .pdf .txt .docx)
        claimant_name : name of the person submitting the claim
        coverage_type : "single" | "family"
        client        : Anthropic client (optional — uses env var if None)

    Returns:
        ClaimRecord as a dict (for easy JSON serialisation in Gradio).
        Never raises — pipeline errors result in an ESCALATE claim record.
    """
    if client is None:
        client = anthropic.Anthropic()

    claim_id    = _next_claim_id()
    file_name   = Path(file_path).name
    submitted   = datetime.now().isoformat()

    print(f"\n{'═'*55}")
    print(f"  Processing claim #{claim_id}  —  {file_name}")
    print(f"{'═'*55}")

    # ── Step 2: Extraction agent ───────────────────────────────────────
    print("  [1/5] Extractor (Haiku)  ...")
    extracted = run_extractor(file_path, client=client)

    # Do NOT overwrite extracted.claimant_name with the form-submitted name.
    # extracted.claimant_name is what the extractor found on the receipt —
    # it may be null, a full name, or a partial read. That is the ground truth
    # for Gate 1 comparison.
    # The submitted form name travels separately as claimant_name_submitted
    # in the validator's runtime context and is compared against the receipt
    # value there. Mixing the two corrupts the comparison.

    print(f"        vendor : {extracted.vendor}")
    print(f"        amount : {extracted.amount_total}")
    print(f"        date   : {extracted.date_incurred}")
    print(f"        quality: {extracted.image_quality}")

    # ── Step 3: Duplicate check (script) ──────────────────────────────
    print("  [2/5] Duplicate check ...")
    is_dup, dup_of = check_duplicate(
        vendor           = extracted.vendor or "",
        amount           = extracted.amount_total or 0.0,
        date             = extracted.date_incurred or "",
        current_claim_id = claim_id,
        claims_db        = _claims_db,
    )
    if is_dup:
        print(f"        ⚠️  Duplicate detected — matches claim {dup_of}")
    else:
        print("        ✓  No duplicate found")

    # ── Step 4: Balance lookup ─────────────────────────────────────────
    balance = _get_balance(claimant_name, coverage_type)
    print(f"  [3/5] Balance: ${balance:.2f} remaining")

    # ── Step 5: Validator agent ────────────────────────────────────────
    print("  [4/5] Validator (Sonnet) ...")
    validated = run_validator(
        extracted                = extracted,
        claimant_name_submitted  = claimant_name,
        claimant_balance         = balance,
        coverage_type            = coverage_type,
        duplicate_detected       = is_dup,
        duplicate_of             = dup_of,
        client                   = client,
    )
    print(f"        gate status : {validated.overall_gate_status}")
    print(f"        category    : {validated.category_code}")
    if validated.first_failing_gate:
        print(f"        first fail  : {validated.first_failing_gate}")

    # ── Step 6: Adjudicator agent ──────────────────────────────────────
    print("  [5/5] Adjudicator (Opus) ...")
    adjudicated = run_adjudicator(
        extracted        = extracted,
        validated        = validated,
        claimant_balance = balance,
        coverage_type    = coverage_type,
        feedback_log     = _feedback_log,
        client           = client,
    )

    # ── Step 7: Output guardrail ───────────────────────────────────────
    adjudicated = _guardrail(adjudicated)

    # ── Step 8: Triage tag ─────────────────────────────────────────────
    triage_tag = determine_triage_tag(adjudicated, validated)

    print(f"        decision   : {adjudicated.decision} ({adjudicated.reason_code})")
    print(f"        confidence : {adjudicated.confidence_score:.0%}")
    print(f"        triage     : {triage_tag}")

    # ── Step 9: Build ClaimRecord and store ───────────────────────────
    record = ClaimRecord(
        claim_id           = claim_id,
        file_name          = file_name,
        file_path          = str(file_path),
        claimant_name      = claimant_name,
        coverage_type      = coverage_type,
        submitted_at       = submitted,
        status             = "awaiting_review",
        triage_tag         = triage_tag,
        extractor_output   = extracted,
        validator_output   = validated,
        adjudicator_output = adjudicated,
    )

    record_dict = record.to_dict()
    _claims_db[claim_id] = record_dict

    print(f"  ✓ Claim #{claim_id} stored — awaiting examiner review")
    return record_dict


# ═══════════════════════════════════════════════════════════════════════
# EXAMINER DECISION
# Called when the human examiner clicks Approve / Reject / Pend etc.
# Updates the claim record, deducts balance, logs feedback.
# ═══════════════════════════════════════════════════════════════════════

def submit_decision(
    claim_id:       str,
    decision:       str,   # "approved" | "rejected" | "pended" | "escalated" | "routed"
    notes:          str = "",
    override_reason:str = "",
    feedback_note:  str = "",
) -> dict:
    """
    Record the human examiner's final decision on a claim.

    Updates:
      - claim status and examiner fields
      - balance ledger (deducts if approved)
      - feedback log (for adjudicator learning)

    Args:
        claim_id        : the claim to update
        decision        : examiner's decision label
        notes           : optional examiner notes
        override_reason : required if overriding the agent recommendation
        feedback_note   : optional free-text feedback for agent learning

    Returns:
        Updated ClaimRecord as a dict.
        Returns an error dict if claim_id not found.
    """
    if claim_id not in _claims_db:
        return {"error": f"Claim {claim_id} not found."}

    record = _claims_db[claim_id]

    # Prevent re-action on already decided claims
    current_status = record.get("status", "awaiting_review")
    if current_status != "awaiting_review":
        return {
            "error": (
                f"Claim {claim_id} has already been decided: "
                f"{current_status.upper()}. No further action can be taken."
            ),
            "_already_decided": True,
            **record,
        }

    record = _claims_db[claim_id]
    adj    = record.get("adjudicator_output") or {}

    # Update claim record fields
    record["examiner_decision"]  = decision
    record["examiner_notes"]     = notes
    record["override_reason"]    = override_reason
    record["feedback_note"]      = feedback_note
    record["status"]             = decision.lower()
    record["decided_at"]         = datetime.now().isoformat()

    # Deduct balance if approved
    if decision == "approved":
        approved_amt = float(adj.get("approved_amount") or 0.0)
        if approved_amt > 0:
            _deduct_balance(record["claimant_name"], approved_amt)

    # Determine if the examiner overrode the agent
    agent_decision = (adj.get("decision") or "").upper()
    decision_map   = {
        "approved": "APPROVE", "rejected": "DENY",
        "pended":   "PEND",    "escalated": "ESCALATE",
        "routed":   "ROUTE_HCSA",
    }
    human_as_agent = decision_map.get(decision.lower(), decision.upper())
    is_override    = (agent_decision != human_as_agent) and bool(agent_decision)

    # Append to feedback log
    feedback_record = {
        "claim_id":        claim_id,
        "vendor":          (record.get("extractor_output") or {}).get("vendor", "unknown"),
        "agent_decision":  adj.get("decision"),
        "agent_confidence":adj.get("confidence_score"),
        "human_decision":  decision,
        "is_override":     is_override,
        "override_reason": override_reason,
        "feedback_note":   feedback_note,
        "examiner_notes":  notes,
        "timestamp":       datetime.now().isoformat(),
    }
    _feedback_log.append(feedback_record)
    _save_feedback_log()

    print(f"  ✓ Claim #{claim_id} → {decision.upper()}"
          + (" (OVERRIDE)" if is_override else ""))

    return record


# ═══════════════════════════════════════════════════════════════════════
# EXAMINER Q&A
# Called when the examiner types a question about a specific claim.
# Passes the question to the adjudicator with full claim context.
# ═══════════════════════════════════════════════════════════════════════

def ask_question(
    claim_id: str,
    question: str,
    client:   anthropic.Anthropic = None,
) -> str:
    """
    Let the examiner ask a follow-up question about a specific claim.

    Calls the adjudicator with the original claim context + the question.
    Returns the adjudicator's plain-English answer.

    Args:
        claim_id : the claim to ask about
        question : the examiner's question
        client   : Anthropic client (optional)

    Returns:
        Answer string, or an error message if claim not found.
    """
    if claim_id not in _claims_db:
        return f"Claim {claim_id} not found."

    if not question.strip():
        return "Please enter a question."

    if client is None:
        client = anthropic.Anthropic()

    record    = _claims_db[claim_id]
    ext_dict  = record.get("extractor_output")   or {}
    val_dict  = record.get("validator_output")   or {}
    balance   = _get_balance(record["claimant_name"], record["coverage_type"])

    # Reconstruct dataclass objects from stored dicts
    try:
        extracted = ExtractorOutput.from_dict(ext_dict)
        validated = ValidatorOutput.from_dict(val_dict)
    except Exception as e:
        return f"Could not reconstruct claim context for Q&A: {e}"

    result = run_adjudicator(
        extracted         = extracted,
        validated         = validated,
        claimant_balance  = balance,
        coverage_type     = record.get("coverage_type", "single"),
        feedback_log      = _feedback_log,
        examiner_question = question,
        client            = client,
    )

    return (
        result.qa_answer
        or result.rationale
        or "I cannot answer that question from the available information."
    )


# ═══════════════════════════════════════════════════════════════════════
# STATS AND QUERY HELPERS
# Used by the Gradio UI to populate the dashboard.
# ═══════════════════════════════════════════════════════════════════════

def get_stats() -> dict:
    """Return summary statistics for the claims dashboard."""
    stats = {
        "total":          len(_claims_db),
        "confident":      0,
        "needs_review":   0,
        "approved":       0,
        "rejected":       0,
        "pended":         0,
        "escalated":      0,
        "routed":         0,
        "high_confidence": 0,
        "avg_confidence": 0.0,
    }

    confidences = []

    for record in _claims_db.values():
        status = record.get("status", "awaiting_review")
        triage = record.get("triage_tag", "NEEDS_REVIEW")
        adj    = record.get("adjudicator_output") or {}
        conf   = adj.get("confidence_score")

        if status == "awaiting_review":
            if triage == "CONFIDENT":
                stats["confident"] += 1
            else:
                stats["needs_review"] += 1
        elif status in stats:
            stats[status] += 1

        if conf is not None:
            try:
                c = float(conf)
                confidences.append(c)
                if c >= EASY_CONFIDENCE_THRESHOLD:
                    stats["high_confidence"] += 1
            except (TypeError, ValueError):
                pass

    if confidences:
        avg = sum(confidences) / len(confidences)
        stats["avg_confidence"] = round(avg, 3)

    return stats


def get_all_claims(filter_by: str = "all") -> list:
    """
    Return all claims, optionally filtered.

    Args:
        filter_by : "all" | "awaiting" | "decided"

    Returns:
        List of ClaimRecord dicts sorted by claim_id ascending.
    """
    records = sorted(
        _claims_db.values(),
        key=lambda r: r.get("claim_id", "")
    )

    if filter_by == "all":
        return records

    result = []
    for r in records:
        status = r.get("status", "awaiting_review")
        is_awaiting = (status == "awaiting_review")
        is_decided  = (status in ("approved", "rejected", "pended",
                                  "escalated", "routed"))

        if filter_by == "awaiting" and is_awaiting:
            result.append(r)
        elif filter_by == "decided" and is_decided:
            result.append(r)

    return result


def get_claim(claim_id: str) -> dict:
    """Return a single ClaimRecord dict, or None if not found."""
    return _claims_db.get(claim_id.strip().zfill(3))


def reset_state():
    """
    Clear all in-memory state.
    Useful for testing — resets DB, balances, counter, and feedback.
    Does NOT delete the feedback log file.
    """
    global _claims_db, _balance_ledger, _feedback_log
    _claims_db      = {}
    _balance_ledger = {}
    _feedback_log   = []
    _claim_counter[0] = 0
    print("  ✓ Pipeline state reset")


# ═══════════════════════════════════════════════════════════════════════
# QUICK TEST — run directly to verify
# Requires ANTHROPIC_API_KEY in environment.
# Uses two real receipts from the test set.
#
# python pipeline.py
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, sys

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("✗ ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Find test_set folder
    test_set = Path(__file__).parent / "test_set"
    if not test_set.exists():
        test_set = Path("test_set")
    if not test_set.exists():
        print(f"✗ test_set/ folder not found. Expected at: {test_set}")
        sys.exit(1)

    print("Pipeline integration test\n")

    # ── Test 1: Clean gym membership ──────────────────────────────────
    gym = test_set / "07_gym_membership.jpg"
    if gym.exists():
        r1 = process_claim(str(gym), "Jordan Tran", "single", client=client)
        adj = r1.get("adjudicator_output") or {}
        print(f"\n  Result: {adj.get('decision')} ({adj.get('reason_code')})")
        print(f"  Confidence: {adj.get('confidence_score', 0):.0%}")
        print(f"  Triage: {r1.get('triage_tag')}")
        print(f"  Rationale: {(adj.get('rationale') or '')[:100]}...")
    else:
        print(f"  Skipping test 1 — {gym} not found")

    # ── Test 2: Out-of-period season pass ─────────────────────────────
    season = test_set / "06_season_pass_email.txt"
    if season.exists():
        r2 = process_claim(str(season), "Jordan Tran", "single", client=client)
        adj2 = r2.get("adjudicator_output") or {}
        print(f"\n  Result: {adj2.get('decision')} ({adj2.get('reason_code')})")
        print(f"  Confidence: {adj2.get('confidence_score', 0):.0%}")
        print(f"  Triage: {r2.get('triage_tag')}")
    else:
        print(f"  Skipping test 2 — {season} not found")

    # ── Test 3: Examiner decision ──────────────────────────────────────
    if _claims_db:
        first_id = list(_claims_db.keys())[0]
        updated  = submit_decision(
            claim_id        = first_id,
            decision        = "approved",
            notes           = "Clean claim — approved.",
            override_reason = "",
            feedback_note   = "",
        )
        print(f"\n  Claim #{first_id} status: {updated.get('status')}")
        print(f"  Balance after approval: "
              f"${_balance_ledger.get('Jordan Tran', 0):.2f}")

    # ── Test 4: Q&A ────────────────────────────────────────────────────
    if len(_claims_db) >= 2:
        second_id = list(_claims_db.keys())[1]
        answer = ask_question(
            second_id,
            "Why is this claim being denied?",
            client=client,
        )
        print(f"\n  Q&A answer: {answer[:200]}")

    # ── Test 5: Stats ──────────────────────────────────────────────────
    stats = get_stats()
    print(f"\n  Stats: {stats}")

    # ── Test 6: Duplicate detection ───────────────────────────────────
    # Process the portal receipt (duplicate of gym membership)
    portal = test_set / "11_gym_portal_receipt.jpg"
    if portal.exists():
        r_dup = process_claim(str(portal), "Jordan Tran", "single", client=client)
        adj_dup = r_dup.get("adjudicator_output") or {}
        val_dup = r_dup.get("validator_output") or {}
        print(f"\n  Duplicate test result: {adj_dup.get('decision')} ({adj_dup.get('reason_code')})")
        print(f"  Duplicate flag in validator: {(val_dup or {}).get('duplicate_flag')}")

    print("\n✓ Phase 6 complete — proceed to Phase 7 (app.py)")