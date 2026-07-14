"""
adjudicator.py
──────────────
Adjudication Agent — Claude Opus 4.8

Responsibility:
  Reason over ExtractorOutput + ValidatorOutput and produce a
  recommendation for the human examiner. Collaborative tone.
  Explicit confidence score with shown working.
  Never raises — escalates gracefully on failure.

Input:
  ExtractorOutput    — from the Extraction Agent
  ValidatorOutput    — from the Validation Agent
  claimant_balance   — current remaining WSA balance
  coverage_type      — "single" | "family"
  feedback_context   — recent examiner overrides (for learning)
  examiner_question  — optional follow-up question from the examiner

Output:
  AdjudicatorOutput dataclass (see schemas.py)

Usage:
  from agents.models.adjudicator import run_adjudicator
  result = run_adjudicator(extracted, validated, balance=750.0)
"""

import json
import re
from pathlib import Path

import anthropic

try:
    from agents.schemas import (
        ExtractorOutput, ValidatorOutput,
        AdjudicatorOutput, PLAN, REASON_CODES
    )
except ImportError:
    from schemas import (
        ExtractorOutput, ValidatorOutput,
        AdjudicatorOutput, PLAN, REASON_CODES
    )

# ── Model ──────────────────────────────────────────────────────────────
MODEL = "claude-opus-4-8"

# ── System prompt ──────────────────────────────────────────────────────
_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "adjudicator_system.txt"

def _load_system_prompt() -> str:
    if _PROMPT_PATH.exists():
        print("Adjudicator: Read prompt file successfully.")
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are a WSA claims adjudication agent. Reason over the "
        "provided extractor and validator outputs. Produce a recommendation "
        "with a confidence score. Collaborative tone. Return only valid JSON."
    )

SYSTEM_PROMPT = _load_system_prompt()


# ═══════════════════════════════════════════════════════════════════════
# PART 1 — FEEDBACK CONTEXT BUILDER
# Reads recent examiner overrides and formats them for the prompt.
# This is how the adjudicator learns from the examiner over time.
# ═══════════════════════════════════════════════════════════════════════

def build_feedback_context(feedback_log: list, max_entries: int = 10) -> str:
    """
    Format recent examiner override reasons and feedback notes for the adjudicator.

    IMPORTANT: This function passes ONLY explicit override reasons and feedback
    notes written by the examiner. It does NOT pass actual approve/deny decisions,
    agent recommendations, or any information about what happened to other claims.
    The adjudicator must reason from the manual and booklet only — not from
    inferring patterns of past decisions.

    Args:
        feedback_log:  list of feedback dicts from the claims database
        max_entries:   how many recent entries to include

    Returns:
        A formatted string of examiner corrections and notes only.
    """
    if not feedback_log:
        return "No examiner feedback from prior claims yet."

    recent = feedback_log[-max_entries:]
    lines = ["EXAMINER CORRECTIONS AND FEEDBACK NOTES:"]
    lines.append(
        "The following are explicit notes left by the examiner on prior claims.\n"
        "Use these only to understand how the examiner interprets edge cases.\n"
        "Do NOT infer approval or denial patterns from these notes.\n"
    )

    useful_entries = 0
    for entry in recent:
        vendor   = entry.get("vendor",          "unknown vendor")
        note     = entry.get("feedback_note",   "").strip()
        override = entry.get("override_reason", "").strip()

        # Only pass explicit examiner-written text — nothing else
        if not note and not override:
            continue

        line = "- Regarding a claim from " + vendor + ":"
        if override:
            line += "\n  Examiner correction: " + override
        if note:
            line += "\n  Examiner feedback note: " + note
        lines.append(line)
        useful_entries += 1

    if useful_entries == 0:
        return (
            "No override feedback yet. "
            "Apply the manual and booklet rules as written."
        )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# PART 2 — PROMPT BUILDER
# Builds the user message with all inputs concatenated cleanly.
# No triple-quoted f-strings — explicit concatenation only.
# ═══════════════════════════════════════════════════════════════════════

def _build_user_message(
    extracted: ExtractorOutput,
    validated: ValidatorOutput,
    claimant_balance: float,
    coverage_type: str,
    feedback_context: str,
    examiner_question: str = None,
) -> str:
    """
    Build the user message for the adjudicator.
    All inputs are concatenated as plain strings.
    """
    balance_str    = "$" + f"{claimant_balance:.2f}"
    allocation     = (
        "$" + str(int(PLAN["family_allocation"]))
        if coverage_type == "family"
        else "$" + str(int(PLAN["single_allocation"]))
    )

    # Reason codes block — gives Opus the full vocabulary to choose from
    codes_block = "\n".join(
        "  " + code + "  — " + desc
        for code, desc in REASON_CODES.items()
    )

    # Optional Q&A instruction
    qa_instruction = ""
    if examiner_question and examiner_question.strip():
        qa_instruction = (
            "\n\nEXAMINER QUESTION:\n"
            + examiner_question.strip()
            + "\nAnswer this in the qa_answer field. "
            "Cite the relevant rule. Be concise."
        )

    return (
        "Please adjudicate the following WSA claim.\n\n"

        "─── EXTRACTOR OUTPUT ───────────────────────────────────────\n"
        + extracted.to_json()
        + "\n\n"

        "─── VALIDATOR OUTPUT ───────────────────────────────────────\n"
        + validated.to_json()
        + "\n\n"

        "─── RUNTIME CONTEXT ────────────────────────────────────────\n"
        "  claimant_balance  : " + balance_str + "\n"
        "  coverage_type     : " + coverage_type + "\n"
        "  annual_allocation : " + allocation + "\n"
        "  plan_year         : " + PLAN["year_start"] + " to " + PLAN["year_end"] + "\n"
        "  runout_deadline   : " + PLAN["runout_deadline"] + "\n\n"

        "─── AVAILABLE REASON CODES ─────────────────────────────────\n"
        + codes_block
        + "\n\n"

        "─── EXAMINER FEEDBACK (recent corrections) ─────────────────\n"
        + feedback_context
        + qa_instruction
        + "\n\n"

        "Reason over the extractor and validator outputs.\n"
        "Produce your recommendation with full confidence score working.\n"
        "Use collaborative, suggestive tone — not assertive.\n"
        "Return only valid JSON matching the AdjudicatorOutput schema."
    )


# ═══════════════════════════════════════════════════════════════════════
# PART 3 — JSON PARSER
# Identical safe parser used across all three agents.
# ═══════════════════════════════════════════════════════════════════════

def _parse_json_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return {"_parse_error": True, "_raw": text[:500]}


# ═══════════════════════════════════════════════════════════════════════
# PART 4 — RESPONSE MAPPER
# Maps raw JSON dict → AdjudicatorOutput dataclass.
# Applies two hard rules regardless of what Opus returned:
#   Rule 1: Every APPROVE must have tax_flag = True
#   Rule 2: confidence_score must be between 0.05 and 1.0
# ═══════════════════════════════════════════════════════════════════════

_TAX_NOTE = (
    "WSA reimbursement is a taxable employment benefit under the Income Tax Act. "
    "The approved amount must be reported to payroll for source deductions "
    "(income tax, CPP, EI). For Quebec members, QPP and QPIP also apply; "
    "expect RL-1 reporting in addition to the federal T4. See Manual §4.2."
)

def _apply_hard_rules(parsed: dict) -> dict:
    """
    Enforce hard rules that must hold regardless of LLM output.
    Mutates and returns the dict.
    """
    decision = (parsed.get("decision") or "").upper()

    # Rule 1: Tax flag on every approval
    if decision in ("APPROVE",):
        if not parsed.get("tax_flag"):
            parsed["tax_flag"] = True
            parsed["tax_note"] = _TAX_NOTE

    # Rule 2: Confidence score bounds
    score = parsed.get("confidence_score")
    if score is not None:
        try:
            parsed["confidence_score"] = max(0.05, min(1.0, float(score)))
        except (TypeError, ValueError):
            parsed["confidence_score"] = 0.50
    else:
        parsed["confidence_score"] = 0.50

    # Rule 3: Ensure reason_code is present
    if not parsed.get("reason_code"):
        # Make a best guess from the decision
        fallback_codes = {
            "APPROVE":     "A-OK",
            "DENY":        "D-EXP",
            "PEND":        "P-DOC",
            "ROUTE_HCSA":  "R-HCSA",
            "ESCALATE":    "P-REV",
        }
        parsed["reason_code"] = fallback_codes.get(decision, "P-REV")

    return parsed


def _map_to_adjudicator_output(parsed: dict) -> AdjudicatorOutput:
    """Map raw parsed JSON → AdjudicatorOutput after applying hard rules."""
    parsed = _apply_hard_rules(parsed)

    return AdjudicatorOutput(
        decision              = parsed.get("decision")              or "ESCALATE",
        reason_code           = parsed.get("reason_code")           or "P-REV",
        approved_amount       = float(parsed.get("approved_amount") or 0.0),
        denied_amount         = float(parsed.get("denied_amount")   or 0.0),
        confidence_score      = parsed["confidence_score"],
        confidence_reasoning  = parsed.get("confidence_reasoning")  or "",
        rationale             = parsed.get("rationale")             or "",
        cited_rules           = parsed.get("cited_rules")           or [],
        tax_flag              = bool(parsed.get("tax_flag",  False)),
        tax_note              = parsed.get("tax_note"),
        examiner_suggestions  = parsed.get("examiner_suggestions")  or "",
        uncertainty_notes     = parsed.get("uncertainty_notes"),
        i_dont_know           = bool(parsed.get("i_dont_know", False)),
        qa_answer             = parsed.get("qa_answer"),
        feedback_incorporated = parsed.get("feedback_incorporated"),
    )


# ═══════════════════════════════════════════════════════════════════════
# PART 5 — TRIAGE TAGGER
# After the adjudicator returns, determine whether the claim goes into
# the EASY queue or the REVIEW queue for the human examiner.
#
# EASY   → High confidence + clean gates + approve/deny recommended
#           Examiner can action quickly with one click.
# REVIEW → Lower confidence, flags, unusual circumstances, or pend/escalate
#           Examiner needs to read carefully before acting.
# ═══════════════════════════════════════════════════════════════════════

EASY_CONFIDENCE_THRESHOLD = 0.80  # Minimum confidence to qualify as EASY

def determine_triage_tag(
    adjudicator_out: AdjudicatorOutput,
    validator_out:   ValidatorOutput,
) -> str:
    """
    Determine queue tag for a claim.

    CONFIDENT  — agent made a clear APPROVE or DENY with confidence >= 80%,
                 no flags, no uncertainty. Examiner can act quickly.
    NEEDS_REVIEW — agent could not make a confident recommendation,
                 confidence is below 80%, something is missing or flagged,
                 or decision is PEND/ESCALATE/ROUTE_HCSA.
    """
    decision    = (adjudicator_out.decision or "").upper()
    confidence  = adjudicator_out.confidence_score or 0.0
    i_dont_know = adjudicator_out.i_dont_know
    dup_flag    = validator_out.duplicate_flag if validator_out else False

    if decision not in ("APPROVE", "DENY"):
        return "NEEDS_REVIEW"

    if confidence < EASY_CONFIDENCE_THRESHOLD:
        return "NEEDS_REVIEW"

    if i_dont_know:
        return "NEEDS_REVIEW"

    if dup_flag:
        return "NEEDS_REVIEW"

    return "CONFIDENT"


# ═══════════════════════════════════════════════════════════════════════
# PART 6 — ADJUDICATOR AGENT
# Main entry point. Calls Opus 4.8.
# ═══════════════════════════════════════════════════════════════════════

def run_adjudicator(
    extracted:          ExtractorOutput,
    validated:          ValidatorOutput,
    claimant_balance:   float,
    coverage_type:      str  = "single",
    feedback_log:       list = None,
    examiner_question:  str  = None,
    client:             anthropic.Anthropic = None,
) -> AdjudicatorOutput:
    """
    Run the Adjudication Agent on extractor + validator outputs.

    Args:
        extracted         : ExtractorOutput from Extraction Agent
        validated         : ValidatorOutput from Validation Agent
        claimant_balance  : remaining WSA balance for this claimant
        coverage_type     : "single" | "family"
        feedback_log      : list of prior feedback dicts (can be empty)
        examiner_question : optional follow-up question from the examiner
        client            : Anthropic client. If None, uses ANTHROPIC_API_KEY env var.

    Returns:
        AdjudicatorOutput with recommendation, confidence, and rationale.
        Never raises — returns a safe ESCALATE output on any failure.
    """
    if client is None:
        client = anthropic.Anthropic()

    feedback_context = build_feedback_context(feedback_log or [])

    # ── Step 1: Build the user message ────────────────────────────────
    user_message = _build_user_message(
        extracted          = extracted,
        validated          = validated,
        claimant_balance   = claimant_balance,
        coverage_type      = coverage_type,
        feedback_context   = feedback_context,
        examiner_question  = examiner_question,
    )

    # ── Step 2: Call Opus 4.8 ─────────────────────────────────────────
    try:
        response = client.messages.create(
            model      = MODEL,
            max_tokens = 2048,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_message}],
        )
        raw_text = response.content[0].text

    except Exception as e:
        return AdjudicatorOutput(
            decision             = "ESCALATE",
            reason_code          = "P-REV",
            confidence_score     = 0.05,
            confidence_reasoning = "Adjudicator API call failed — escalating for manual review.",
            rationale            = (
                "I was unable to process this claim due to a technical error. "
                "I'd recommend escalating to a senior examiner rather than "
                "making a decision without my analysis. Error: " + str(e)
            ),
            examiner_suggestions = "Please review this claim manually.",
            i_dont_know          = True,
        )

    # ── Step 3: Parse JSON ─────────────────────────────────────────────
    parsed = _parse_json_response(raw_text)

    if parsed.get("_parse_error"):
        return AdjudicatorOutput(
            decision             = "ESCALATE",
            reason_code          = "P-REV",
            confidence_score     = 0.05,
            confidence_reasoning = "Could not parse adjudicator response.",
            rationale            = (
                "I produced a response that couldn't be parsed as structured data. "
                "Rather than guess at a decision, I'd suggest escalating this one "
                "for manual review. Raw output snippet: "
                + (parsed.get("_raw") or "")
            ),
            examiner_suggestions = "Please review this claim manually.",
            i_dont_know          = True,
        )

    # ── Step 4: Map to AdjudicatorOutput + apply hard rules ───────────
    return _map_to_adjudicator_output(parsed)


# ═══════════════════════════════════════════════════════════════════════
# QUICK TEST — run directly to verify
# Requires ANTHROPIC_API_KEY in environment.
#
# python agents/models/adjudicator.py
#
# Tests three scenarios:
#   1. Clean approval  (GoodLife gym — expect APPROVE, high confidence)
#   2. Out of period   (2023 season pass — expect DENY D-PER)
#   3. Medical expense (physiotherapy — expect ROUTE_HCSA)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, sys

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("✗ ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("Testing adjudicator...\n")

    # ── Shared extracted fact for tests 1 & 2 ─────────────────────────
    gym_extracted = ExtractorOutput(
        vendor="GoodLife Fitness", date_incurred="2025-03-14",
        amount_pretax=150.00, amount_total=169.50, tax_amount=19.50,
        items=["Membership - 6 month"], claimant_name="Jordan Tran",
        payment_confirmed=True, payment_method="Visa ****4127",
        image_quality="good", document_type="receipt",
    )

    # ── Test 1: Clean approval ─────────────────────────────────────────
    from agents.schemas import ValidatorOutput, GateResult, BalanceCheck

    clean_validated = ValidatorOutput(
        gate1_eligible_person  = GateResult(True,  "Jordan Tran — eligible. Note: not verified against membership system.", "Booklet — Who Is Covered"),
        gate2_eligible_expense = GateResult(True,  "Gym membership — WSA-FIT.", "Manual §9.1"),
        gate2_in_period        = GateResult(True,  "2025-03-14 within benefit year 2025.", "Manual §5.2"),
        gate3_documentation    = GateResult(True,  "All required fields present. PAID IN FULL.", "Manual §8.1"),
        balance_check          = BalanceCheck(750.00, 169.50, True, None),
        category_code          = "WSA-FIT",
        overall_gate_status    = "PASS",
        validator_notes        = "Clean claim. No flags.",
    )

    print("Test 1 — GoodLife Fitness (expect APPROVE, confidence >= 0.80)")
    r1 = run_adjudicator(gym_extracted, clean_validated, 750.00, "single", client=client)
    print("  decision             :", r1.decision)
    print("  reason_code          :", r1.reason_code)
    print("  approved_amount      :", r1.approved_amount)
    print("  confidence_score     :", r1.confidence_score)
    print("  tax_flag             :", r1.tax_flag)
    print("  triage_tag           :", determine_triage_tag(r1, clean_validated))
    print("  rationale            :", r1.rationale[:120] + "...")
    print("  confidence_reasoning :", r1.confidence_reasoning[:120] + "...")
    print()

    # ── Test 2: Out-of-period deny ─────────────────────────────────────
    old_extracted = ExtractorOutput(
        vendor="Blue Mountain Resort", date_incurred="2023-11-14",
        amount_total=597.77, items=["Winter Season Pass 2023/24"],
        claimant_name="Jordan Tran", payment_confirmed=True,
        image_quality="good", document_type="email",
    )
    old_validated = ValidatorOutput(
        gate1_eligible_person  = GateResult(True,  "Name matches. Eligibility assumed.", "Booklet — Who Is Covered"),
        gate2_eligible_expense = GateResult(True,  "Season ski pass — WSA-FIT.", "Manual §9.1"),
        gate2_in_period        = GateResult(False, "Date 2023-11-14 is outside 2025 benefit year.", "Manual §5.2"),
        gate3_documentation    = GateResult(True,  "Email receipt with all required fields.", "Manual §8.1"),
        balance_check          = BalanceCheck(750.00, 597.77, True, None),
        category_code          = "WSA-FIT",
        overall_gate_status    = "FAIL",
        first_failing_gate     = "gate2_period",
        validator_notes        = "Expense is from 2023 — two years outside the benefit year.",
    )

    print("Test 2 — 2023 season pass (expect DENY D-PER)")
    r2 = run_adjudicator(old_extracted, old_validated, 750.00, "single", client=client)
    print("  decision     :", r2.decision)
    print("  reason_code  :", r2.reason_code)
    print("  confidence   :", r2.confidence_score)
    print("  triage_tag   :", determine_triage_tag(r2, old_validated))
    print("  rationale    :", r2.rationale[:120] + "...")
    print()

    # ── Test 3: Medical expense routing ───────────────────────────────
    physio_extracted = ExtractorOutput(
        vendor="Riverside Physiotherapy Clinic", date_incurred="2025-09-12",
        amount_total=245.00, tax_amount=0.00,
        items=["Physiotherapy assessment", "Follow-up physiotherapy (treatment of injury)"],
        claimant_name="Jordan Tran", payment_confirmed=True,
        image_quality="good", document_type="invoice",
        extraction_notes="Referral from Dr. A. Pereira (MD). Dx: Post-op knee rehab. Zero HST — consistent with medical service.",
    )
    physio_validated = ValidatorOutput(
        gate1_eligible_person  = GateResult(True,  "Jordan Tran — eligible. Eligibility assumed.", "Booklet"),
        gate2_eligible_expense = GateResult(False, "Post-op physiotherapy is a CRA medical expense.", "Manual §9 — HCSA routing"),
        gate2_in_period        = GateResult(True,  "2025-09-12 within benefit year.", "Manual §5.2"),
        gate3_documentation    = GateResult(True,  "Invoice with all required fields. Paid in full.", "Manual §8.1"),
        balance_check          = BalanceCheck(750.00, 245.00, True, None),
        is_medical_expense     = True,
        route_to_hcsa          = True,
        overall_gate_status    = "HCSA",
        first_failing_gate     = "gate2_expense",
        validator_notes        = "Physiotherapy treating post-surgical condition. Route to HCSA.",
    )

    print("Test 3 — Physiotherapy (expect ROUTE_HCSA)")
    r3 = run_adjudicator(physio_extracted, physio_validated, 750.00, "single", client=client)
    print("  decision     :", r3.decision)
    print("  reason_code  :", r3.reason_code)
    print("  confidence   :", r3.confidence_score)
    print("  triage_tag   :", determine_triage_tag(r3, physio_validated))
    print("  rationale    :", r3.rationale[:120] + "...")
    print()

    # ── Test 4: Q&A from examiner (no new API agent — same call) ──────
    print("Test 4 — Q&A: 'Why is physiotherapy routed to HCSA not WSA?'")
    r4 = run_adjudicator(
        physio_extracted, physio_validated, 750.00, "single",
        examiner_question="Why is physiotherapy routed to HCSA and not paid from the WSA?",
        client=client,
    )
    print("  qa_answer:", (r4.qa_answer or "None")[:200])
    print()

    # ── Test 5: Serialisation ──────────────────────────────────────────
    d = r1.to_dict()
    assert isinstance(d, dict)
    assert "decision" in d
    assert "confidence_score" in d
    assert r1.tax_flag == True, "Tax flag must be set on APPROVE"
    print("  ✓ AdjudicatorOutput serialises correctly")
    print("  ✓ Tax flag enforced on approval")
    print()
    print("✓ Phase 5 complete — proceed to Phase 6 (pipeline.py)")