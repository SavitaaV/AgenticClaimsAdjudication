"""
validator.py
────────────
Validation Agent — Claude Sonnet 4.6

Responsibility:
  Take ExtractorOutput and verify the claim against the WSA plan rules.
  Run the three-gate test in order. Check balance. Flag duplicates.
  Surface findings. Make NO final decision.

Input:
  ExtractorOutput   — from the Extraction Agent
  claimant_balance  — current remaining WSA balance (float)
  coverage_type     — "single" | "family"
  duplicate_detected — True if same vendor+amount+date found in batch

Output:
  ValidatorOutput dataclass (see schemas.py)
  — gate results with findings and citations
  — no approve/deny — findings only

Usage:
  from agents.models.validator import run_validator
  result = run_validator(extractor_output, claimant_name_submitted="Jordan Tran", claimant_balance=750.0, coverage_type="single")
"""

import json
import re
from pathlib import Path

import anthropic

try:
    from agents.schemas import (
        ExtractorOutput, ValidatorOutput,
        GateResult, BalanceCheck, PLAN
    )
except ImportError:
    from schemas import (
        ExtractorOutput, ValidatorOutput,
        GateResult, BalanceCheck, PLAN
    )

# ── Model ──────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"

# ── System prompt ──────────────────────────────────────────────────────
_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "validator_system.txt"

def _load_system_prompt() -> str:
    if _PROMPT_PATH.exists():
        print("Validator: Read prompt file successfully.")
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are a WSA claims validator. Run the three-gate test. "
        "Report findings only. Make no decision. Return only valid JSON."
    )

SYSTEM_PROMPT = _load_system_prompt()


# ═══════════════════════════════════════════════════════════════════════
# PART 1 — PROMPT BUILDER
# Injects runtime context (balance, coverage, duplicate flag)
# into the user message alongside extracted facts.
# The system prompt has the rules. The user message has the data.
# ═══════════════════════════════════════════════════════════════════════

def _build_user_message(
    extracted: ExtractorOutput,
    claimant_name_submitted: str,
    claimant_balance: float,
    coverage_type: str,
    duplicate_detected: bool,
    duplicate_of: str = None,
) -> str:
    # Extract all values into plain variables first.
    # Avoids triple-quote f-string issues across editors and Python versions.
    facts          = extracted.to_json()
    submitted_name = claimant_name_submitted or "not provided"
    balance_str    = f"${claimant_balance:.2f}"
    year_start     = PLAN["year_start"]
    year_end       = PLAN["year_end"]
    runout         = PLAN["runout_deadline"]

    duplicate_note = ""
    if duplicate_detected:
        ref = f" (matches claim {duplicate_of})" if duplicate_of else ""
        duplicate_note = (
            "\nDUPLICATE FLAG: YES — same vendor, amount, and date "
            "found in another claim in this batch" + ref + ". "
            "Set duplicate_flag: true in your response."
        )

    return (
        "Please validate the following WSA claim.\n\n"
        "EXTRACTED CLAIM FACTS:\n"
        + facts +
        "\n\nRUNTIME CONTEXT:\n"
        "  claimant_name_submitted : " + submitted_name + "\n"
        "  claimant_balance        : " + balance_str + "\n"
        "  coverage_type           : " + coverage_type + "\n"
        "  plan_year               : " + year_start + " to " + year_end + "\n"
        "  runout_deadline         : " + runout
        + duplicate_note +
        "\n\nRun the three-gate test in strict order.\n"
        "Surface your findings for each gate with citations.\n"
        "Do NOT approve or deny — report findings only.\n"
        "Return only valid JSON matching the ValidatorOutput schema."
    )


# ═══════════════════════════════════════════════════════════════════════
# PART 2 — JSON PARSER
# Same safe parser as extractor — handles markdown fences.
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
# PART 3 — RESPONSE MAPPER
# Maps the raw JSON dict from Sonnet → ValidatorOutput dataclass.
# Defensive: every field uses .get() so missing keys never crash.
# ═══════════════════════════════════════════════════════════════════════

def _map_gate_result(d: dict) -> GateResult:
    """Safely map a gate dict → GateResult."""
    if not d:
        return GateResult(
            passed   = False,
            finding  = "Gate result missing from validator response.",
            citation = None,
        )
    return GateResult(
        passed   = bool(d.get("passed", False)),
        finding  = d.get("finding")  or "No finding provided.",
        citation = d.get("citation"),
    )


def _map_balance_check(d: dict, fallback_balance: float, fallback_amount: float) -> BalanceCheck:
    """Safely map a balance_check dict → BalanceCheck."""
    if not d:
        return BalanceCheck(
            available_balance = fallback_balance,
            claim_amount      = fallback_amount,
            sufficient        = fallback_balance >= fallback_amount,
            partial_amount    = fallback_balance if fallback_balance < fallback_amount else None,
        )
    return BalanceCheck(
        available_balance = float(d.get("available_balance") or fallback_balance),
        claim_amount      = float(d.get("claim_amount")      or fallback_amount),
        sufficient        = bool(d.get("sufficient", False)),
        partial_amount    = d.get("partial_amount"),
    )


def _map_to_validator_output(
    parsed: dict,
    fallback_balance: float,
    fallback_amount: float,
) -> ValidatorOutput:
    """Map raw parsed JSON → ValidatorOutput dataclass."""
    return ValidatorOutput(
        gate1_eligible_person  = _map_gate_result(parsed.get("gate1_eligible_person")),
        gate2_eligible_expense = _map_gate_result(parsed.get("gate2_eligible_expense")),
        gate2_in_period        = _map_gate_result(parsed.get("gate2_in_period")),
        gate3_documentation    = _map_gate_result(parsed.get("gate3_documentation")),
        balance_check          = _map_balance_check(
                                    parsed.get("balance_check"),
                                    fallback_balance,
                                    fallback_amount,
                                 ),
        is_medical_expense     = bool(parsed.get("is_medical_expense", False)),
        route_to_hcsa          = bool(parsed.get("route_to_hcsa",      False)),
        category_code          = parsed.get("category_code"),
        duplicate_flag         = bool(parsed.get("duplicate_flag",     False)),
        duplicate_of           = parsed.get("duplicate_of"),
        overall_gate_status    = parsed.get("overall_gate_status")  or "FAIL",
        first_failing_gate     = parsed.get("first_failing_gate"),
        validator_notes        = parsed.get("validator_notes")      or "",
    )


# ═══════════════════════════════════════════════════════════════════════
# PART 4 — DUPLICATE CHECKER (script — no LLM needed)
# Called by the pipeline before run_validator.
# Checks the current claims batch for same vendor + amount + date.
# ═══════════════════════════════════════════════════════════════════════

def check_duplicate(
    vendor: str,
    amount: float,
    date: str,
    current_claim_id: str,
    claims_db: dict,
) -> tuple[bool, str]:
    """
    Check whether this claim duplicates any existing claim in the batch.

    Args:
        vendor          : vendor name from extraction
        amount          : total amount from extraction
        date            : date incurred from extraction
        current_claim_id: ID of the claim being checked (exclude from search)
        claims_db       : the in-memory claims database

    Returns:
        (is_duplicate: bool, duplicate_of: str)
        duplicate_of is the claim_id of the matching claim, or "" if none.
    """
    if not vendor or not amount or not date:
        return False, ""

    vendor_clean = (vendor or "").strip().lower()

    for cid, claim in claims_db.items():
        if cid == current_claim_id:
            continue

        ex = claim.get("extractor_output")
        if not ex:
            continue

        # Support both dict (from DB) and ExtractorOutput object
        if isinstance(ex, dict):
            v = (ex.get("vendor") or "").strip().lower()
            a = ex.get("amount_total") or 0
            d = ex.get("date_incurred") or ""
        else:
            v = (ex.vendor or "").strip().lower()
            a = ex.amount_total or 0
            d = ex.date_incurred or ""

        if (v == vendor_clean
                and abs(a - amount) < 0.01
                and d == date):
            return True, cid

    return False, ""


# ═══════════════════════════════════════════════════════════════════════
# PART 5 — VALIDATOR AGENT
# Main entry point. Calls Sonnet 4.6.
# ═══════════════════════════════════════════════════════════════════════

def run_validator(
    extracted: ExtractorOutput,
    claimant_name_submitted: str,
    claimant_balance: float,
    coverage_type: str       = "single",
    duplicate_detected: bool = False,
    duplicate_of: str        = None,
    client: anthropic.Anthropic = None,
) -> ValidatorOutput:
    """
    Run the Validation Agent on extracted claim facts.

    Args:
        extracted         : ExtractorOutput from the Extraction Agent
        claimant_balance  : remaining WSA balance for this claimant
        coverage_type     : "single" | "family"
        duplicate_detected: True if a duplicate was found in the batch
        duplicate_of      : claim ID of the duplicate match, if any
        client            : Anthropic client. If None, uses ANTHROPIC_API_KEY env var.

    Returns:
        ValidatorOutput with gate findings, balance check, and flags.
        Never raises — returns a safe error output on failure.
    """
    if client is None:
        client = anthropic.Anthropic()

    # ── Step 1: Build the user message ────────────────────────────────
    user_message = _build_user_message(
        extracted                = extracted,
        claimant_name_submitted  = claimant_name_submitted,
        claimant_balance         = claimant_balance,
        coverage_type            = coverage_type,
        duplicate_detected       = duplicate_detected,
        duplicate_of             = duplicate_of,
    )

    # ── Step 2: Call Sonnet 4.6 ───────────────────────────────────────
    try:
        response = client.messages.create(
            model      = MODEL,
            max_tokens = 2048,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_message}],
        )
        raw_text = response.content[0].text

    except Exception as e:
        # API failure — return a safe escalate output
        return ValidatorOutput(
            overall_gate_status = "PENDING",
            first_failing_gate  = None,
            validator_notes     = f"Validator API call failed: {e}. Escalating for manual review.",
            duplicate_flag      = duplicate_detected,
            duplicate_of        = duplicate_of,
        )

    # ── Step 3: Parse JSON ─────────────────────────────────────────────
    parsed = _parse_json_response(raw_text)

    if parsed.get("_parse_error"):
        return ValidatorOutput(
            overall_gate_status = "PENDING",
            validator_notes     = (
                f"Could not parse validator response. "
                f"Raw output: {parsed.get('_raw', '')}. "
                f"Escalating for manual review."
            ),
            duplicate_flag = duplicate_detected,
            duplicate_of   = duplicate_of,
        )

    # ── Step 4: Map to ValidatorOutput ────────────────────────────────
    fallback_amount = extracted.amount_total or 0.0

    return _map_to_validator_output(
        parsed           = parsed,
        fallback_balance = claimant_balance,
        fallback_amount  = fallback_amount,
    )


# ═══════════════════════════════════════════════════════════════════════
# QUICK TEST — run directly to verify
# Requires ANTHROPIC_API_KEY in environment.
#
# python agents/models/validator.py
#
# Tests two claims:
#   Claim 07 (GoodLife gym) — expect PASS
#   Claim 06 (2023 season pass) — expect FAIL gate2_in_period
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, sys

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("✗ ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("Testing validator...\n")

    # ── Test 1: Clean claim — expect PASS ─────────────────────────────
    clean = ExtractorOutput(
        vendor            = "GoodLife Fitness",
        date_incurred     = "2025-03-14",
        amount_pretax     = 150.00,
        amount_total      = 169.50,
        tax_amount        = 19.50,
        items             = ["Membership - 6 month"],
        claimant_name     = "Jordan Tran",
        payment_confirmed = True,
        payment_method    = "Visa ****4127",
        image_quality     = "good",
        document_type     = "receipt",
    )

    print("Test 1 — GoodLife Fitness gym membership (expect PASS)")
    result1 = run_validator(
        extracted                = clean,
        claimant_name_submitted  = "Jordan Tran",
        claimant_balance         = 750.00,
        coverage_type            = "single",
        duplicate_detected       = False,
        client                   = client,
    )
    print(f"  overall_gate_status    : {result1.overall_gate_status}")
    print(f"  category_code          : {result1.category_code}")
    print(f"  gate1 passed           : {result1.gate1_eligible_person.passed if result1.gate1_eligible_person else 'missing'}")
    print(f"  gate2_expense passed   : {result1.gate2_eligible_expense.passed if result1.gate2_eligible_expense else 'missing'}")
    print(f"  gate2_period passed    : {result1.gate2_in_period.passed if result1.gate2_in_period else 'missing'}")
    print(f"  gate3 passed           : {result1.gate3_documentation.passed if result1.gate3_documentation else 'missing'}")
    print(f"  balance sufficient     : {result1.balance_check.sufficient if result1.balance_check else 'missing'}")
    print(f"  first_failing_gate     : {result1.first_failing_gate}")
    print()

    # ── Test 2: Out-of-period claim — expect FAIL gate2_in_period ─────
    old = ExtractorOutput(
        vendor            = "Blue Mountain Resort",
        date_incurred     = "2023-11-14",
        amount_pretax     = 529.00,
        amount_total      = 597.77,
        tax_amount        = 68.77,
        items             = ["Winter Season Pass 2023/24"],
        claimant_name     = "Jordan Tran",
        payment_confirmed = True,
        payment_method    = "Visa ****4127",
        image_quality     = "good",
        document_type     = "email",
    )

    print("Test 2 — 2023 season pass (expect FAIL — out of period)")
    result2 = run_validator(
        extracted                = old,
        claimant_name_submitted  = "Jordan Tran",
        claimant_balance         = 750.00,
        coverage_type            = "single",
        duplicate_detected       = False,
        client                   = client,
    )
    print(f"  overall_gate_status    : {result2.overall_gate_status}")
    print(f"  gate2_period passed    : {result2.gate2_in_period.passed if result2.gate2_in_period else 'missing'}")
    print(f"  gate2_period finding   : {result2.gate2_in_period.finding if result2.gate2_in_period else 'missing'}")
    print(f"  first_failing_gate     : {result2.first_failing_gate}")
    print()

    # ── Test 3: Duplicate flag ─────────────────────────────────────────
    print("Test 3 — Duplicate check (script — no API call)")
    fake_db = {
        "007": {
            "extractor_output": {
                "vendor":        "GoodLife Fitness",
                "amount_total":  169.50,
                "date_incurred": "2025-03-14",
            }
        }
    }
    is_dup, dup_of = check_duplicate(
        vendor           = "GoodLife Fitness",
        amount           = 169.50,
        date             = "2025-03-14",
        current_claim_id = "011",
        claims_db        = fake_db,
    )
    print(f"  is_duplicate: {is_dup}  (expect True)")
    print(f"  duplicate_of: {dup_of}  (expect 007)")
    print()

    # ── Serialisation check ────────────────────────────────────────────
    d = result1.to_dict()
    assert isinstance(d, dict),               "to_dict() must return dict"
    assert "overall_gate_status" in d,        "overall_gate_status must be present"
    assert "gate1_eligible_person" in d,      "gate1 must be present"
    assert isinstance(d["gate1_eligible_person"], dict), "gate1 must serialise to dict"
    print("  ✓ Serialises correctly")
    print()
    print("✓ Phase 4 complete — proceed to Phase 5 (adjudicator.py)")