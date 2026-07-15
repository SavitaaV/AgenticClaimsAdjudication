# WSA Claims Adjudication Agent

A production-grade, multi-agent AI pipeline for adjudicating Wellness Spending Account (WSA) claims — built with a three-agent Claude architecture, structured prompt compilation from policy documents, and a human-in-the-loop Gradio examiner interface.

---

## What This Is

This prototype automates the first-pass adjudication of WSA benefit claims using a sequential three-agent pipeline. Each agent is narrowed to one cognitive task:

| Agent | Model | Task |
|-------|-------|------|
| **Extractor** | Claude Haiku | Reads the receipt (image, PDF, email, text) and extracts structured fields — no judgment, facts only |
| **Validator** | Claude Sonnet | Runs a three-gate eligibility test against the plan rules and reference documents — surfaces findings, makes no decision |
| **Adjudicator** | Claude Opus | Reasons over the validator's findings, produces a recommendation with confidence score, reason code, and examiner tip |

A human examiner reviews every claim through the Gradio UI, can override the agent recommendation, must leave a mandatory audit note, and can provide feedback that feeds the learning loop.

---

## Why Three Agents Instead of One

Breaking the task across focused agents outperforms a single heavyweight model on multi-step workflows. Anthropic's own engineering findings (June 2025) showed a multi-agent system using Opus as the orchestrator and Sonnet subagents outperformed single-agent Opus by over 90% on their research evaluations. The principle: match the model to the cognitive demand of each step, rather than asking one model to do everything.

In this system:
- The extractor does not need reasoning capability — it needs vision and pattern recognition
- The validator does not make decisions — it applies rules and surfaces facts
- The adjudicator does not re-read receipts — it reasons over structured findings

This separation also makes the system auditable: if the adjudicator gets something wrong, you can trace exactly which gate finding it misread.

---

## Architecture

```
Receipt (image / PDF / email / text)
        │
        ▼
┌───────────────────┐
│   EXTRACTOR       │  Claude Haiku
│   Vision + parse  │  → ExtractorOutput (structured fields)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   VALIDATOR       │  Claude Sonnet
│   Three-gate test │  Reads: validator_system.txt
│   + reference docs│         + Manual.docx + Booklet.docx
└───────────────────┘  → ValidatorOutput (gate findings, no decision)
        │
        ▼
┌───────────────────┐
│   ADJUDICATOR     │  Claude Opus
│   Reason + decide │  Reads: adjudicator_system.txt
└───────────────────┘  → AdjudicatorOutput (decision, confidence, reason code)
        │
        ▼
┌───────────────────┐
│   HUMAN EXAMINER  │  Gradio UI
│   Review, override│  Mandatory notes + audit trail
│   feedback loop   │
└───────────────────┘
```

### Three-Gate Adjudication Logic

- **Gate 1** — Eligible person with active coverage? → `D-PERS` if not
- **Gate 2a** — Expense maps to an active WSA category? → `D-EXP` or `R-HCSA` if not
- **Gate 2b** — Incurred within the benefit period? → `D-PER` if not
- **Gate 3** — Valid itemised receipt with proof of payment and claimant name? → `P-DOC` if not
- **Balance check** — Sufficient credit available? → `A-PART` (partial) or `D-BAL` if not
- **Pass all gates** → `A-OK`

### Reason Codes

| Code | Meaning |
|------|---------|
| `A-OK` | Approved in full |
| `A-PART` | Approved partial — balance insufficient |
| `D-PERS` | Denied — ineligible person |
| `D-EXP` | Denied — ineligible expense |
| `D-PER` | Denied — outside benefit period |
| `D-DOC` | Denied — documentation cannot be cured |
| `D-BAL` | Denied — balance exhausted |
| `P-DOC` | Pended — documentation required |
| `P-REV` | Pended — duplicate, requires review |
| `R-HCSA` | Routed to HCSA — CRA medical expense |

---

## Repo Structure

```
AgenticClaimsAdjudication/
└── wellness-claims-agentic-pipeline/
    ├── agents/
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── extractor.py       # Haiku — receipt parsing
    │   │   ├── validator.py       # Sonnet — three-gate test
    │   │   └── adjudicator.py     # Opus  — decision + confidence
    │   ├── __init__.py
    │   └── schemas.py             # Data contracts, PLAN config, reason codes
    ├── configs/
    │   ├── Wellness_Spending_Account_Adjudication_Manual.docx
    │   └── Wellness_Spending_Account_Group_Benefit_Booklet.docx
    ├── prompts/
    │   ├── extractor_system.txt
    │   ├── validator_system.txt
    │   └── adjudicator_system.txt
    ├── test_set/                  # Empty — add your own receipts here
    ├── __init__.py
    ├── app.py                     # Gradio examiner UI
    ├── create_agent_prompts.py    # Utility — compile prompts from reference docs
    ├── pipeline.py                # Orchestration — runs the three-agent sequence
    └── requirements.txt
```

> **Note:** `test_set/` is intentionally empty. Add your own receipt files (JPG, PNG, PDF, TXT, DOCX) to test the pipeline. The system accepts any mix of formats in a single batch.

---

## Prerequisites

- A Google account with Google Drive (for Colab)
- An [Anthropic API key](https://console.anthropic.com/) — create one at console.anthropic.com
- The repo files uploaded to Google Drive in the same folder structure shown above

---

## Setup — Google Colab

### Step 1 — Upload the repo to Google Drive

Upload the entire `AgenticClaimsAdjudication/` folder to the root of your Google Drive, preserving the folder structure exactly as shown in the repo.

### Step 2 — Open the Colab notebook

A ready-to-run Colab notebook is included in the repo root:

**`WellnessClaimPorcessingAgenticWorkflow.ipynb`**

Open it directly in Google Colab — click the file in GitHub and then click **Open in Colab**, or go to [colab.research.google.com](https://colab.research.google.com) → File → Open notebook → GitHub → paste the repo URL. The notebook already contains all the setup cells in the right order — you do not need to create a new notebook or copy any commands manually.

> **Before running:** Add your Anthropic API key to Colab Secrets. Click the 🔑 icon in the left sidebar → New secret → Name: `ANTHROPIC_API_KEY`, Value: your key. Do not paste your key directly into a cell.

### Step 3 — Mount Google Drive and set the project path

Paste each block below into a **separate Colab cell** and run them in order. Do not proceed to the next cell until the current one completes without errors.

**Cell 1-A — Mount Drive and set project path**
```python
from google.colab import drive, userdata
drive.mount('/content/drive')

import os
PROJECT_FOLDER = '/content/drive/MyDrive/AgenticClaimsAdjudication/wellness-claims-agentic-pipeline'
```

### Step 4 — Install dependencies

Run each install command in its own cell in this order:

**Cell 1-B — Core packages**
```python
!pip install -q google-generativeai Pillow pymupdf python-docx pydantic python-docx pypdf
```

**Cell 1-C — Gradio (pinned version)**
```python
!pip install -q gradio==6.19.0
```

**Cell 1-D — Anthropic SDK**
```python
!pip install -q anthropic
```

Expected output after each cell: download progress bars with no red errors.

### Step 5 — Launch the Gradio UI

**Cell 2 — Run the app**
```python
from google.colab import userdata

# Retrieve API key from Colab Secrets
ANTHROPIC_API_KEY_VALUE = userdata.get('ANTHROPIC_API_KEY')

# Launch app with ANTHROPIC_API_KEY and PYTHONPATH set for the subprocess
!ANTHROPIC_API_KEY="{ANTHROPIC_API_KEY_VALUE}" PYTHONPATH="{PROJECT_FOLDER}" python "{PROJECT_FOLDER}/app.py"
```

Gradio will print a public share link (e.g. `https://xxxxx.gradio.live`). Open it in any browser. The link is active for 72 hours per session.

---

## Adding Your Own Test Receipts

Drop any receipt files into the `test_set/` folder in Google Drive:

```
test_set/
├── my_gym_receipt.jpg
├── physio_invoice.pdf
├── course_confirmation.txt
└── ...
```

Supported formats: **JPG, PNG, PDF, TXT, DOCX, EML**

The pipeline accepts images (scanned or photographed receipts), PDFs (including scanned), plain text exports, and email confirmations. The extractor handles format detection automatically.

---

## Using the Examiner UI

The Gradio interface has two tabs:

**Tab 1 — Submit a claim**
- Upload a receipt file
- Enter the claimant name as it appears on the submission form
- Select coverage type (single / family)
- Click Submit — the three-agent pipeline runs automatically

**Tab 2 — Claims queue**
- Filter by status: All / Needs Review / Confident / Decided
- Click View on any claim to open the detail view
- Review the agent recommendation, gate analysis, and confidence score
- Fill in the mandatory examiner notes
- Approve, Partially Approve, Deny, Pend, or Escalate
- If overriding the agent, an override reason is required
- Optionally leave feedback for the agent learning loop

---

## Configuration

All plan configuration lives in one place — `agents/schemas.py`:

```python
PLAN = {
    "benefit_year":      "2025",
    "year_start":        "2025-01-01",
    "year_end":          "2025-12-31",
    "runout_deadline":   "2026-03-31",   # 90 days after year end
    "single_allocation": 750.00,
    "family_allocation": 1000.00,
    "reimbursement_pct": 1.0,            # 100%
}
```

Change the plan year, allocations, or runout deadline here and it propagates everywhere — no other file needs to change.

### Key Configuration Levers

| What to change | Where | How |
|----------------|-------|-----|
| Plan year / allocations | `agents/schemas.py` | Edit the `PLAN` block |
| Confidence threshold for auto-triage | `agents/models/adjudicator.py` | Change `EASY_CONFIDENCE_THRESHOLD = 0.80` |
| Add a new WSA category | `prompts/validator_system.txt` | Add entry in the WSA categories section |
| Add a new reason code | `agents/schemas.py` | Add to `REASON_CODES` dict |
| Change agent tone | `prompts/adjudicator_system.txt` | Edit the tone section |
| Add a pre-processing rule | `pipeline.py` | Add before `run_extractor()` call |
| Add a post-processing rule | `pipeline.py` | Add after `run_adjudicator()` call |

---

## Swapping Models

Each agent has a single `MODEL` constant at the top of its file. Change it to use any Anthropic model, or adapt the API call pattern to use any other provider.

```python
# agents/models/extractor.py
MODEL = "claude-haiku-4-5-20251001"   # fast, vision-capable, cost-efficient

# agents/models/validator.py
MODEL = "claude-sonnet-4-6"           # strong instruction-following

# agents/models/adjudicator.py
MODEL = "claude-opus-4-8"             # highest reasoning quality
```

### Using a Different Provider

The API call in each agent follows this pattern:

```python
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
response = client.messages.create(
    model=MODEL,
    max_tokens=2048,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_message}]
)
```

To swap to OpenAI, Gemini, or a local model (Ollama, LM Studio), replace this block with the equivalent client call for that provider and ensure the response parsing reflects that provider's output format. The rest of the pipeline — schemas, orchestration, UI — does not need to change.

---

## Updating Policy Documents

The validator reads both reference documents at import time from the `configs/` folder. To update the policy:

1. Replace the `.docx` files in `configs/` with the updated documents
2. Restart the runtime — the validator will load the new documents automatically

To also regenerate the compiled system prompts from the updated documents:

```python
!python create_agent_prompts.py
```

---

## UI Walkthrough

> **To use the screenshots below:** Create a `screenshots/` folder in the repo root and save each screenshot with the filename shown. GitHub will render them inline in this README.

---

### Step 1 — App launches in Colab

After running the launch cell, Colab prints a public Gradio URL. Open it in any browser.

![Colab launch](screenshots/01_colab_launch.png)

You will see output confirming all three agents read their prompt files successfully, followed by the public URL — something like `https://xxxxx.gradio.live`. The link is valid for up to one week.

---

### Step 2 — Submit a claim

The app opens on the **Submit a claim** tab.

![Submit a claim — empty](screenshots/02_submit_claim.png)

To submit a claim:
1. Click the upload area or drag and drop a receipt file (JPG, PNG, PDF, TXT, or DOCX)
2. Enter the claimant's name exactly as it appears on the submission form
3. Select coverage type — **single** or **family**
4. Click **Submit claim →**

![Submit a claim — filled in](screenshots/04_submit_with_receipt.png)

The status line below the button shows `⏳ Claim submission request is being processed...` while the three-agent pipeline runs. This takes approximately 10–30 seconds depending on the model and receipt complexity.

![Claim submitted successfully](screenshots/05_claim_submitted.png)

When complete you will see `✅ Claim submitted successfully — Your claim has been received and is queued for examiner review.`

---

### Step 3 — View the claims queue

Click the **Claims queue** tab.

![Claims queue — empty](screenshots/03_claims_queue_empty.png)

The stats bar at the top shows a live count of every claim by status — total, confident, needs review, approved, rejected, pended, escalated, routed, high confidence, and average confidence across the batch. Click **Refresh queue** to update after new submissions.

After submitting a claim you will see it appear in the table:

![Claims queue — populated](screenshots/06_claims_queue_populated.png)

The table columns are:

| Column | What it shows |
|--------|--------------|
| ID | Claim number — e.g. `#001` |
| Vendor | Extracted vendor name from the receipt |
| Claimant | Name entered at submission |
| Date | Date incurred, extracted from the receipt |
| Amount | Total claim amount |
| Queue | `🏅 Confident` or `👁 Needs review` |
| Confidence | Agent confidence score as a percentage |
| Code | Agent suggested reason code (e.g. A-OK, P-DOC, D-EXP) |
| Status | Current examiner status (Awaiting / Approved / Rejected etc.) |

Use the filter pills — **All**, **Awaiting Review**, **Decided** — to narrow the view.

---

### Step 4 — Open and review a claim

To open a specific claim for detailed review:

1. Type the claim ID number into the input box above the table — e.g. type `001` for claim `#001`
2. Click **View claim →**

![Claim detail — agent recommendation](screenshots/07_claim_detail_recommendation.png)

The detail view has two panels.

**Left panel — Agent analysis:**
- Agent recommendation with reason code (e.g. `✅ APPROVE — A-OK`)
- Confidence score with expandable breakdown showing each deduction
- Rationale — the adjudicator's reasoning in plain language
- Cited rules — specific manual sections and booklet clauses referenced
- Tax flag — payroll note on every approval (mandatory per Manual §4.2)
- Uncertainty — any caveats the agent flagged
- Examiner tip — a specific action suggested to close any open question
- Claim details table — all extracted fields from the receipt

**Right panel — Examiner decision:**
- Ask the agent — a live Q&A chatbot for questions about this specific claim
- Notes (mandatory) — required for all decisions, forms the audit trail
- Override reason (mandatory if overriding) — required when your decision differs from the agent
- Five action buttons: **Approve**, **Reject**, **Pend**, **Escalate**, **Route to HCSA**
- Feedback for agent learning (optional) — what the agent got right or wrong

---

### Step 5 — View the receipt

Click **🧾 View receipt** to open the original uploaded file alongside the claim analysis.

![Claim detail — receipt view](screenshots/08_claim_detail_receipt.png)

The receipt renders on the left. The examiner decision panel remains visible on the right so you can review the original document and take action without switching views. Click **✕ Close receipt** to return to the full analysis view.

---

### Step 6 — Take an examiner action

![Claim detail — examiner decision](screenshots/09_claim_detail_examiner.png)

To action a claim:

1. Read the agent recommendation, rationale, and examiner tip on the left panel
2. Fill in the mandatory **Notes** field — this is your audit record for the decision
3. If your decision differs from the agent recommendation, fill in the **Override reason** field
4. Click one of the five action buttons:
   - **✅ Approve** — approve in full at the agent's recommended amount
   - **❌ Reject** — deny the claim
   - **⏸ Pend** — hold for additional documentation or information
   - **⚠️ Escalate** — escalate to senior examiner
   - **🔵 Route to HCSA** — redirect to the Health Care Spending Account
5. Optionally leave feedback in the **Feedback for agent learning** field

Once actioned, the claim status updates in the queue and **the claim is locked** — a decided claim cannot be re-actioned from the UI.

---

## Prototype Limitations

This is a research prototype, not a production system. Known limitations:

- **In-memory state** — claims and balances reset on runtime restart. A production deployment would use a persistent database.
- **No live balance API** — every claimant starts at their full annual allocation. A production system would call a membership API to fetch real-time balances and prior claims history.
- **No authentication** — the Gradio UI has no login layer. Production would require SSO or equivalent.
- **Single-session duplicate detection** — duplicates are only caught within the current session batch, not across historical claims.
- **Colab environment** — `app.py` is configured for Colab with `share=True`. For a persistent deployment, run on a server with a stable URL.

---

## Dependencies

```
anthropic              # Anthropic API client
gradio==6.19.0         # Examiner UI (pinned)
Pillow                 # Image handling (JPG, PNG receipts)
python-docx            # Reference document loading (.docx)
pymupdf                # PDF handling (fitz)
pydantic               # Data validation
pypdf                  # PDF utility
google-generativeai    # Google AI SDK (optional — for model swapping)
```

Install individually in Colab as shown in the setup steps above, or all at once locally:

```bash
pip install anthropic "gradio==6.19.0" Pillow python-docx pymupdf pydantic pypdf google-generativeai
```

---

## License

MIT License

Copyright (c) 2026 Savitaa Venkateswaran

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Acknowledgements

Built as part of a technical case study for the AI Builder, Strategy & Transformation role at Canada Life (2026). Reference documents (adjudication manual and benefit booklet) are illustrative and provided as part of the case materials — they are not real Canada Life policy documents.

Research references informing the multi-agent architecture:
- Anthropic Engineering Blog, June 2025 — multi-agent systems with Opus + Sonnet subagents
- Liu et al., Amazon/UIUC, arXiv:2510.11588 — policy document degradation in LLM agents
- Du et al., ICML 2024 — multi-agent debate outperforms single-agent chain-of-thought
