"""
extractor.py
────────────
Extraction Agent — Claude Haiku 4.5

Responsibility:
  Read a receipt in any supported format and extract structured facts.
  No rules. No eligibility judgment. No decision.

Supported input formats:
  .jpg  .jpeg  .png   → image (passed to Haiku as base64 vision)
  .pdf               → text extracted, or converted to image if text sparse
  .txt  .eml         → plain text
  .docx              → text extracted via python-docx

Output:
  ExtractorOutput dataclass (see schemas.py)

Usage:
  from agents.models.extractor import run_extractor
  result = run_extractor("path/to/receipt.jpg")
  print(result.vendor, result.amount_total)
"""

import base64
import json
import re
from pathlib import Path

import anthropic
import fitz                     # PyMuPDF — PDF handling
from docx import Document       # python-docx — .docx handling
from PIL import Image           # Pillow — image handling
import io

# ── Import schema from parent package ─────────────────────────────────
# Adjust import path if running standalone vs. as part of the package
try:
    from agents.schemas import ExtractorOutput
except ImportError:
    from schemas import ExtractorOutput     # fallback for direct execution

# ── Model ──────────────────────────────────────────────────────────────
MODEL = "claude-haiku-4-5-20251001"

# ── Load system prompt ─────────────────────────────────────────────────
# Looks for the prompt file relative to this file's location
_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "extractor_system.txt"

def _load_system_prompt() -> str:
    """Load extractor system prompt from file."""
    if _PROMPT_PATH.exists():
        print("Extractor: Read prompt file successfully.")
        return _PROMPT_PATH.read_text(encoding="utf-8")
    # Fallback: minimal inline prompt so the agent still runs
    return (
        "You are a receipt extraction agent. Extract claim fields from the receipt. "
        "Return ONLY valid JSON matching the ExtractorOutput schema. "
        "Set fields to null if unreadable. Never guess."
    )

SYSTEM_PROMPT = _load_system_prompt()


# ═══════════════════════════════════════════════════════════════════════
# PART 1 — FILE PARSER
# Converts any supported file format into content Haiku can read.
# Returns a tuple: (content_type, content)
#   content_type  "image_b64" | "text"
#   content       base64 string | plain text string
# ═══════════════════════════════════════════════════════════════════════

# Minimum text length from PDF before we treat it as a scanned image
_PDF_TEXT_MIN_CHARS = 80

# Supported extensions
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_TEXT_EXTS  = {".txt", ".eml"}
_PDF_EXTS   = {".pdf"}
_DOCX_EXTS  = {".docx"}


def _image_to_b64(path: str) -> tuple[str, str]:
    """
    Load an image file and return (media_type, base64_string).
    Converts PNG to JPEG if needed to keep payload smaller.
    """
    ext = Path(path).suffix.lower()
    with open(path, "rb") as f:
        raw = f.read()

    # Re-encode as JPEG to reduce payload size for large images
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return "image/jpeg", b64


def _pdf_to_content(path: str) -> tuple[str, str]:
    """
    Extract content from a PDF.
    Strategy:
      1. Try text extraction — fast and cheap.
      2. If text is too sparse (scanned/image PDF), render first page
         as an image and return as base64 instead.
    """
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    text = text.strip()

    if len(text) >= _PDF_TEXT_MIN_CHARS:
        # Good text extraction — return as text
        return "text", text

    # Sparse text — render first page as image and pass as vision input
    page = doc[0]
    mat  = fitz.Matrix(2.0, 2.0)
    pix  = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("jpeg")
    b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
    # Return "image_b64" so run_extractor treats this as a vision input,
    # consistent with how _image_to_b64() returns image files.
    # Returning "image/jpeg" here was the bug — it caused the base64
    # data to be sent as plain text rather than as an image to Haiku.
    return "image_b64", b64


def _docx_to_text(path: str) -> str:
    """Extract plain text from a .docx file."""
    doc = Document(path)
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(lines)


def parse_file(file_path: str) -> tuple[str, str]:
    """
    Parse any supported receipt file into LLM-readable content.

    Args:
        file_path: path to the receipt file

    Returns:
        (content_type, content) where:
          content_type  "image_b64" | "text"
          content       base64-encoded image string | plain text string

    Raises:
        ValueError if the file format is not supported
        FileNotFoundError if the file does not exist
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Receipt file not found: {file_path}")

    ext = path.suffix.lower()

    if ext in _IMAGE_EXTS:
        media_type, b64 = _image_to_b64(str(path))
        return "image_b64", b64

    elif ext in _PDF_EXTS:
        return _pdf_to_content(str(path))

    elif ext in _TEXT_EXTS:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return "text", text

    elif ext in _DOCX_EXTS:
        text = _docx_to_text(str(path))
        return "text", text

    else:
        supported = _IMAGE_EXTS | _TEXT_EXTS | _PDF_EXTS | _DOCX_EXTS
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported: {', '.join(sorted(supported))}"
        )


# ═══════════════════════════════════════════════════════════════════════
# PART 2 — JSON PARSER
# LLMs sometimes wrap JSON in ```json blocks despite instructions.
# This strips that and parses safely.
# ═══════════════════════════════════════════════════════════════════════

def _parse_json_response(text: str) -> dict:
    """
    Safely parse a JSON response from the LLM.
    Handles markdown code fences that the model may add.
    """
    text = text.strip()

    # Strip ` ```json ... ``` ` or ` ``` ... ``` ` wrappers
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: find the first {...} block in the response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Could not parse — return a safe fallback
    return {
        "_parse_error": True,
        "_raw": text[:500],
    }


# ═══════════════════════════════════════════════════════════════════════
# PART 3 — EXTRACTOR AGENT
# Calls Haiku 4.5 with the parsed file content.
# Returns ExtractorOutput.
# ═══════════════════════════════════════════════════════════════════════

def run_extractor(
    file_path: str,
    client: anthropic.Anthropic = None,
) -> ExtractorOutput:
    """
    Run the Extraction Agent on a single receipt file.

    Args:
        file_path:  path to the receipt (.jpg .jpeg .png .pdf .txt .docx)
        client:     anthropic.Anthropic client instance.
                    If None, creates one using ANTHROPIC_API_KEY env var.

    Returns:
        ExtractorOutput with all fields populated (or null where unreadable).

    Note:
        This function never raises on LLM errors — it returns an
        ExtractorOutput with image_quality="unreadable" and all fields
        null so the pipeline can continue and escalate gracefully.
    """
    if client is None:
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    # ── Step 1: Parse the file ─────────────────────────────────────────
    try:
        content_type, content = parse_file(file_path)
    except (FileNotFoundError, ValueError) as e:
        # File could not be read — return a safe error output
        return ExtractorOutput(
            image_quality   = "unreadable",
            document_type   = "unknown",
            unreadable_fields = ["all"],
            extraction_notes = f"File could not be parsed: {e}",
        )

    # ── Step 2: Build the message ──────────────────────────────────────
    if content_type == "image_b64":
        # Vision input — image sent as base64
        user_content = [
            {
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": "image/jpeg",
                    "data":       content,
                },
            },
            {
                "type": "text",
                "text": "Extract all claim fields from this receipt. Return only valid JSON.",
            },
        ]
    else:
        # Text input — plain text sent in the message
        user_content = (
            f"Extract all claim fields from this receipt.\n\n"
            f"RECEIPT CONTENT:\n{content}\n\n"
            f"Return only valid JSON."
        )

    # ── Step 3: Call Haiku 4.5 ─────────────────────────────────────────
    try:
        response = client.messages.create(
            model      = MODEL,
            max_tokens = 1024,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_content}],
        )
        raw_text = response.content[0].text

    except Exception as e:
        # API call failed — return safe fallback
        return ExtractorOutput(
            image_quality    = "unreadable",
            document_type    = "unknown",
            unreadable_fields = ["all"],
            extraction_notes  = f"Extraction API call failed: {e}",
        )

    # ── Step 4: Parse the JSON response ───────────────────────────────
    parsed = _parse_json_response(raw_text)

    if parsed.get("_parse_error"):
        return ExtractorOutput(
            image_quality    = "unreadable",
            document_type    = "unknown",
            unreadable_fields = ["all"],
            extraction_notes  = f"Could not parse extractor response. Raw: {parsed.get('_raw', '')}",
        )

    # ── Step 5: Map to ExtractorOutput ────────────────────────────────
    # Use .get() with defaults so missing keys don't crash
    return ExtractorOutput(
        vendor             = parsed.get("vendor"),
        date_incurred      = parsed.get("date_incurred"),
        amount_pretax      = parsed.get("amount_pretax"),
        amount_total       = parsed.get("amount_total"),
        tax_amount         = parsed.get("tax_amount"),
        items              = parsed.get("items")              or [],
        claimant_name      = parsed.get("claimant_name"),
        payment_confirmed  = bool(parsed.get("payment_confirmed", False)),
        payment_method     = parsed.get("payment_method"),
        unreadable_fields  = parsed.get("unreadable_fields")  or [],
        image_quality      = parsed.get("image_quality")      or "good",
        document_type      = parsed.get("document_type")      or "receipt",
        extraction_notes   = parsed.get("extraction_notes"),
    )


# ═══════════════════════════════════════════════════════════════════════
# QUICK TEST — run this file directly to verify the extractor works
# Set ANTHROPIC_API_KEY in your environment first.
#
# python agents/models/extractor.py
#
# Expected output: extraction result for each test file
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    import sys

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("✗ ANTHROPIC_API_KEY not set. Export it and re-run.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Test the file parser with a small set (no API call)
    print("Testing file parser (no API call)...\n")

    # Find test_set folder relative to this file
    test_set = Path(__file__).parent.parent.parent / "test_set"

    if not test_set.exists():
        print(f"  ✗ test_set folder not found at {test_set}")
        sys.exit(1)

    test_files = sorted(test_set.iterdir())
    if not test_files:
        print(f"  ✗ No files found in {test_set}")
        sys.exit(1)

    # Test parser only (verify format detection, no API cost)
    for f in test_files:
        if f.suffix.lower() in (_IMAGE_EXTS | _TEXT_EXTS | _PDF_EXTS | _DOCX_EXTS):
            try:
                content_type, content = parse_file(str(f))
                size_kb = len(content) // 1024
                print(f"  ✓ {f.name:<40} → {content_type:<12} ({size_kb} KB)")
            except Exception as e:
                print(f"  ✗ {f.name:<40} → ERROR: {e}")

    print("\nParser test complete.\n")

    # Test full extraction on one receipt (uses API)
    sample = test_set / "10_course_enrollment_email.txt"
    if sample.exists():
        print(f"Testing full extraction on {sample.name}...")
        result = run_extractor(str(sample), client=client)
        print(f"\n  vendor           : {result.vendor}")
        print(f"  date_incurred    : {result.date_incurred}")
        print(f"  amount_total     : {result.amount_total}")
        print(f"  items            : {result.items}")
        print(f"  payment_confirmed: {result.payment_confirmed}")
        print(f"  image_quality    : {result.image_quality}")
        print(f"  unreadable_fields: {result.unreadable_fields}")
        print(f"  extraction_notes : {result.extraction_notes}")
        print()

        # Verify it serialises cleanly (needed for pipeline)
        d = result.to_dict()
        assert isinstance(d, dict), "to_dict() must return a dict"
        assert "vendor" in d,       "vendor key must be present"
        print("  ✓ Serialises correctly")
        print()
        print("✓ Phase 3 complete — proceed to Phase 4 (validator.py)")
    else:
        print(f"  Sample file {sample} not found — skipping API test.")
        print("  Add test files to test_set/ and re-run.")