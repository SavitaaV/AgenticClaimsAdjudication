"""
app.py  —  Maple Shield WSA Claims Adjudicator
Gradio UI with two views in the Claims Queue tab:
  - Queue view  (stats + table, click row to open claim)
  - Detail view (full claim + ask agent + examiner actions)
"""

import os, base64, json
from pathlib import Path
import gradio as gr

try:
    from pipeline import (
        process_claim, submit_decision,
        ask_question, get_stats,
        get_all_claims, get_claim,
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from pipeline import (
        process_claim, submit_decision,
        ask_question, get_stats,
        get_all_claims, get_claim,
    )


# ═══════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _decision_icon(decision: str) -> str:
    return {
        "APPROVE":    "✅",
        "DENY":       "❌",
        "PEND":       "⏸",
        "ESCALATE":   "⚠️",
        "ROUTE_HCSA": "🔵",
    }.get((decision or "").upper(), "❓")


def _gate_icon(passed) -> str:
    return "✅" if passed else "❌"


def _fmt_stats(stats: dict) -> str:
    avg = stats.get("avg_confidence", 0.0)
    avg_str = f"{avg*100:.1f}%"
    return (
        f"**{stats.get('total',0)}** total &nbsp;|&nbsp; "
        f"🏅 **{stats.get('confident',0)}** confident &nbsp;|&nbsp; "
        f"👁 **{stats.get('needs_review',0)}** needs review &nbsp;|&nbsp; "
        f"✅ **{stats.get('approved',0)}** approved &nbsp;|&nbsp; "
        f"❌ **{stats.get('rejected',0)}** rejected &nbsp;|&nbsp; "
        f"⏸ **{stats.get('pended',0)}** pended &nbsp;|&nbsp; "
        f"⚠️ **{stats.get('escalated',0)}** escalated &nbsp;|&nbsp; "
        f"🔵 **{stats.get('routed',0)}** routed &nbsp;|&nbsp; "
        f"🎯 **{stats.get('high_confidence',0)}** high conf. &nbsp;|&nbsp; "
        f"avg conf. **{avg_str}**"
    )


def build_claims_table(filter_by: str = "all") -> list:
    filter_map = {
        "All":             "all",
        "Awaiting Review": "awaiting",
        "Decided":         "decided",
    }
    claims = get_all_claims(filter_by=filter_map.get(filter_by, "all"))
    rows = []
    for c in claims:
        adj  = c.get("adjudicator_output") or {}
        ex   = c.get("extractor_output")   or {}
        tag  = c.get("triage_tag", "NEEDS_REVIEW")
        stat = c.get("status", "awaiting_review")

        tag_label = "🏅 Confident" if tag == "CONFIDENT" else "👁 Needs review"
        conf_str  = f"{(adj.get('confidence_score') or 0)*100:.0f}%"

        status_labels = {
            "awaiting_review": "⏳ Awaiting",
            "approved":        "✅ Approved",
            "rejected":        "❌ Rejected",
            "pended":          "⏸ Pended",
            "escalated":       "⚠️ Escalated",
            "routed":          "🔵 Routed HCSA",
        }
        rows.append([
            f"#{c.get('claim_id','—')}",
            ex.get("vendor")         or "—",
            c.get("claimant_name")   or "—",
            ex.get("date_incurred")  or "—",
            f"${ex.get('amount_total') or 0:.2f}",
            tag_label,
            conf_str,
            adj.get("reason_code")   or "—",
            status_labels.get(stat, stat),
        ])
    return rows


def fmt_claim_header(claim: dict) -> str:
    if not claim:
        return "No claim loaded."
    cid    = claim.get("claim_id", "—")
    ex     = claim.get("extractor_output") or {}
    vendor = ex.get("vendor") or "Unknown vendor"
    status = claim.get("status", "awaiting_review")
    status_labels = {
        "awaiting_review": "Awaiting Review",
        "approved":  "Approved",
        "rejected":  "Rejected",
        "pended":    "Pended",
        "escalated": "Escalated",
        "routed":    "Routed to HCSA",
    }
    status_str = status_labels.get(status, status.replace("_"," ").title())
    return f"## Claim #{cid} — {vendor} &nbsp; *{status_str}*"


def fmt_agent_recommendation(claim: dict) -> str:
    if not claim:
        return ""
    adj  = claim.get("adjudicator_output") or {}

    decision  = adj.get("decision")         or "—"
    code      = adj.get("reason_code")      or "—"
    conf      = adj.get("confidence_score") or 0.0
    conf_pct  = f"{conf*100:.0f}%"
    reasoning = (adj.get("confidence_reasoning") or "No breakdown available.").strip()
    rationale = adj.get("rationale")        or "—"
    rules     = adj.get("cited_rules")      or []
    tax_flag  = adj.get("tax_flag",  False)
    tax_note  = adj.get("tax_note")         or ""
    uncertain = adj.get("uncertainty_notes") or ""
    tip       = adj.get("examiner_suggestions") or ""
    i_dont_know = adj.get("i_dont_know", False)

    icon      = _decision_icon(decision)
    conf_bar  = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
    rules_md  = "\n".join(f"  - {r}" for r in rules) if rules else "  - —"

    # Format confidence reasoning as line-by-line list inside a collapsible block
    # Each sentence becomes its own line for readability
    reasoning_lines = [
        s.strip() for s in reasoning.replace(". ", ".\n").split("\n")
        if s.strip()
    ]
    reasoning_formatted = "\n".join(
        f"> {line}" for line in reasoning_lines
    )

    tax_md      = f"\n\n⚠️ **Tax flag:** {tax_note}" if tax_flag else ""
    uncertain_md = f"\n\n❓ **Uncertainty:** {uncertain}" if uncertain else ""
    tip_md      = f"\n\n💡 **Examiner tip:** {tip}" if tip else ""
    i_dont_know_md = "\n\n🚩 **Agent flagged:** Could not make a confident determination." if i_dont_know else ""

    # Wrap confidence reasoning in HTML <details> so it's hidden by default
    # Examiner clicks the ℹ️ to expand
    confidence_detail = (
        "<details><summary><strong>" + conf_pct + "</strong> &nbsp; ℹ️ <em>How this score was calculated</em></summary>\n\n"
        + reasoning_formatted
        + "\n\n</details>"
    )

    return (
        f"### {icon} {decision} — {code}\n\n"
        f"**Confidence:** {confidence_detail}\n\n"
        f"`{conf_bar}`\n\n"
        f"**Rationale:**\n\n"
        f"*\"{rationale}\"*\n\n"
        f"**Cited rules:**\n{rules_md}"
        f"{tax_md}"
        f"{uncertain_md}"
        f"{i_dont_know_md}"
        f"{tip_md}"
    )


def fmt_claim_details(claim: dict) -> str:
    if not claim:
        return ""
    ex  = claim.get("extractor_output") or {}
    v   = claim.get("validator_output") or {}
    bc  = v.get("balance_check") or {}
    adj = claim.get("adjudicator_output") or {}

    vendor    = ex.get("vendor")          or "—"
    date      = ex.get("date_incurred")   or "—"
    pretax    = ex.get("amount_pretax")
    total     = ex.get("amount_total")
    tax       = ex.get("tax_amount")
    items     = ex.get("items")           or []
    payment   = ex.get("payment_method")  or "—"
    confirmed = ex.get("payment_confirmed", False)
    quality   = ex.get("image_quality")   or "—"
    doc_type  = ex.get("document_type")   or "—"
    unread    = ex.get("unreadable_fields") or []
    ext_notes = ex.get("extraction_notes") or ""

    bal_before   = bc.get("available_balance")
    approved_amt = adj.get("approved_amount") or 0.0
    bal_after    = max(0.0, bal_before - approved_amt) if bal_before is not None else None

    def fmt_money(v):
        return f"${v:.2f}" if v is not None else "—"

    pay_icon     = "✅" if confirmed else "❌"
    quality_icon = {"good": "✅", "poor": "⚠️", "unreadable": "❌"}.get(quality, "—")
    items_str    = ", ".join(items) if items else "—"

    unread_md  = f"\n\n⚠️ **Unreadable fields:** {', '.join(unread)}" if unread else ""
    notes_md   = f"\n\n📝 **Extraction note:** {ext_notes}" if ext_notes else ""

    return (
        f"### 📋 Claim details\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Vendor | {vendor} |\n"
        f"| Date incurred | {date} |\n"
        f"| Pre-tax | {fmt_money(pretax)} |\n"
        f"| Tax | {fmt_money(tax)} |\n"
        f"| **Total paid** | **{fmt_money(total)}** |\n"
        f"| Items | {items_str} |\n"
        f"| Payment | {pay_icon} {payment} |\n"
        f"| Document type | {doc_type} |\n"
        f"| Image quality | {quality_icon} {quality} |\n"
        f"| Coverage | {claim.get('coverage_type','—').title()} |\n"
        f"| Balance before | {fmt_money(bal_before)} |\n"
        f"| Balance if approved | {fmt_money(bal_after)} |"
        f"{unread_md}{notes_md}"
    )


def fmt_gate_analysis(claim: dict) -> str:
    if not claim:
        return ""
    v   = claim.get("validator_output") or {}
    g1  = v.get("gate1_eligible_person")   or {}
    g2e = v.get("gate2_eligible_expense")  or {}
    g2p = v.get("gate2_in_period")         or {}
    g3  = v.get("gate3_documentation")     or {}
    bc  = v.get("balance_check")           or {}
    dup = v.get("duplicate_flag",  False)
    dup_of = v.get("duplicate_of") or ""
    hcsa   = v.get("route_to_hcsa", False)

    def gate_block(icon, title, finding, citation=None):
        cite = f"\n  *{citation}*" if citation else ""
        return f"{icon} **{title}**\n  {finding}{cite}\n"

    g1_assumption = ""
    if g1.get("passed"):
        g1_assumption = " *(eligibility assumed — not verified against membership system)*"

    bal_avail = bc.get("available_balance")
    bal_claim = bc.get("claim_amount")
    bal_suf   = bc.get("sufficient", False)
    bal_icon  = "✅" if bal_suf else "⚠️"
    partial   = bc.get("partial_amount")

    bal_str = "—"
    if bal_avail is not None and bal_claim is not None:
        bal_str = f"${bal_claim:.2f} claimed vs ${bal_avail:.2f} available."
        if not bal_suf and partial is not None:
            bal_str += f" Partial approval possible up to ${partial:.2f}."

    hcsa_md = "\n🔵 **HCSA routing:** This is a CRA medical expense. Submit to HCSA, not WSA.\n" if hcsa else ""
    dup_md  = ""
    if dup:
        ref = f" — matches claim {dup_of}" if dup_of else ""
        dup_md = f"\n⚠️ **Duplicate flag:** Same vendor, amount, and date found in another claim{ref}.\n"

    return (
        "### 🔍 Three-gate adjudication\n\n"
        + gate_block(_gate_icon(g1.get("passed")),  "Gate 1 — eligible person",
                     (g1.get("finding") or "—") + g1_assumption,
                     g1.get("citation"))
        + "\n"
        + gate_block(_gate_icon(g2e.get("passed")), "Gate 2a — eligible expense",
                     g2e.get("finding") or "—",
                     g2e.get("citation"))
        + "\n"
        + gate_block(_gate_icon(g2p.get("passed")), "Gate 2b — in benefit period",
                     g2p.get("finding") or "—",
                     g2p.get("citation"))
        + "\n"
        + gate_block(_gate_icon(g3.get("passed")),  "Gate 3 — valid documentation",
                     g3.get("finding") or "—",
                     g3.get("citation"))
        + f"\n{bal_icon} **Balance check**\n  {bal_str}\n"
        + hcsa_md
        + dup_md
    )


def load_receipt_for_display(claim: dict):
    """
    Load the original receipt file for display.
    Returns (image_or_none, text_or_none, file_type)
    """
    if not claim:
        return None, None, None

    file_path = claim.get("file_path")
    if not file_path or not Path(file_path).exists():
        return None, "Receipt file not found.", "text"

    ext = Path(file_path).suffix.lower()
    try:
        if ext in (".jpg", ".jpeg", ".png"):
            return file_path, None, "image"
        elif ext == ".pdf":
            import fitz
            doc = fitz.open(file_path)
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pix.save(tmp.name)
            return tmp.name, None, "image"
        elif ext in (".txt", ".eml"):
            text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            return None, text, "text"
        else:
            return None, f"Cannot display file type: {ext}", "text"
    except Exception as e:
        return None, f"Error loading receipt: {e}", "text"


# ═══════════════════════════════════════════════════════════════════════
# BUILD APP
# ═══════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:

    with gr.Blocks(
        title="Maple Shield — WSA Adjudicator",
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        /* Centre the submit form */
        .submit-form-col {
            max-width: 520px !important;
            margin: 3rem auto !important;
        }
        /* Clickable table rows */
        .claims-table tbody tr { cursor: pointer; }
        .claims-table tbody tr:hover { background: var(--block-background-fill) !important; }
        /* Receipt viewer */
        .receipt-panel { border: 1px solid var(--border-color-primary); border-radius: 8px; }
        """
    ) as demo:

        gr.Markdown(
            "# 🛡️ Maple Shield — WSA Claims Adjudicator\n"
            "*Wellness Spending Account · AI-assisted adjudication · Human review*"
        )

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════
            # TAB 1 — SUBMIT A CLAIM
            # ══════════════════════════════════════════════════════════
            with gr.Tab("📤 Submit a claim"):
                with gr.Row():
                    with gr.Column(scale=1):
                        pass   # left spacer
                    with gr.Column(scale=2, elem_classes=["submit-form-col"]):
                        gr.Markdown(
                            "### Submit a WSA Claim\n"
                            "Upload your receipt and fill in your details. "
                            "Your claim will be processed and queued for review."
                        )
                        upload_file  = gr.File(
                            label="Receipt",
                            file_types=[".jpg",".jpeg",".png",".pdf",".txt",".docx"],
                            file_count="single",
                        )
                        name_input   = gr.Textbox(
                            label="Your name",
                            placeholder="e.g. Jordan Tran",
                        )
                        coverage_sel = gr.Radio(
                            choices=["single", "family"],
                            value="single",
                            label="Coverage type",
                        )
                        submit_btn   = gr.Button("Submit claim →", variant="primary")
                        submit_out   = gr.Markdown("")
                    with gr.Column(scale=1):
                        pass   # right spacer

                def handle_submit(file, name, cov):
                    if not file:
                        return "⚠️ Please upload a receipt before submitting."
                    if not name or not name.strip():
                        return "⚠️ Please enter your name."
                    try:
                        claim = process_claim(
                            file_path     = file,
                            claimant_name = name.strip(),
                            coverage_type = cov,
                        )
                        return "### ✅ Claim submitted successfully\n\nYour claim has been received and is queued for examiner review."
                    except Exception as e:
                        return f"❌ Error processing claim: {e}"

                def validate_and_show_processing(file, name):
                    """
                    Runs instantly on button click — before the API call.
                    Validates inputs and returns an immediate status message.
                    If inputs are invalid, returns the error so .then() is
                    skipped (Gradio still chains, but handle_submit will
                    re-validate and return the same error — harmless).
                    """
                    if not file:
                        return "⚠️ Please upload a receipt before submitting."
                    if not name or not name.strip():
                        return "⚠️ Please enter your name."
                    return "⏳ Claim submission request is being processed..."

                (
                    submit_btn.click(
                        validate_and_show_processing,
                        inputs=[upload_file, name_input],
                        outputs=[submit_out],
                    ).then(
                        handle_submit,
                        inputs=[upload_file, name_input, coverage_sel],
                        outputs=[submit_out],
                    )
                )

            # ══════════════════════════════════════════════════════════
            # TAB 2 — CLAIMS QUEUE
            # Two views toggled by visibility:
            #   queue_view  — default, shows table
            #   detail_view — shown when a row is clicked
            # ══════════════════════════════════════════════════════════
            with gr.Tab("📋 Claims queue"):

                # ── QUEUE VIEW ────────────────────────────────────────
                with gr.Column(visible=True) as queue_view:

                    stats_md = gr.Markdown(_fmt_stats(get_stats()))

                    with gr.Row():
                        filter_radio = gr.Radio(
                            choices=["All", "Awaiting Review", "Decided"],
                            value="All",
                            label="Filter",
                            interactive=True,
                        )
                        refresh_btn = gr.Button(
                            "🔄 Refresh queue",
                            size="sm",
                            variant="secondary",
                        )

                    # ── Claim viewer — centred between filter and table ──
                    with gr.Row():
                        with gr.Column(scale=1):
                            pass
                        with gr.Column(scale=2):
                            with gr.Row():
                                claim_id_input = gr.Textbox(
                                    label="",
                                    placeholder="Enter claim ID to view  (e.g. 001)",
                                    show_label=False,
                                    scale=3,
                                )
                                view_claim_btn = gr.Button(
                                    "View claim →",
                                    variant="primary",
                                    scale=1,
                                )
                            open_error_md = gr.Markdown("")
                        with gr.Column(scale=1):
                            pass

                    claims_table = gr.Dataframe(
                        headers=[
                            "ID","Vendor","Claimant","Date",
                            "Amount","Queue","Confidence","Code","Status"
                        ],
                        datatype=["str"]*9,
                        value=build_claims_table(),
                        label="Claims",
                        interactive=False,
                        wrap=True,
                        row_count=(15, "dynamic"),
                    )

                # ── DETAIL VIEW ────────────────────────────────────────
                with gr.Column(visible=False) as detail_view:

                    # Back button + header
                    with gr.Row():
                        back_btn      = gr.Button("← Back to queue", size="sm", variant="secondary")
                        detail_header = gr.Markdown("", scale=5)

                    gr.Markdown("---")

                    # Two columns
                    with gr.Row():

                        # ── LEFT COLUMN ───────────────────────────────
                        # 1. Agent recommendation
                        # 2. Claim details
                        # 3. Three-gate adjudication
                        with gr.Column(scale=3):

                            # Receipt toggle button + viewer
                            with gr.Row():
                                view_receipt_btn  = gr.Button("🧾 View receipt", size="sm", variant="secondary")
                                close_receipt_btn = gr.Button("✕ Close receipt", size="sm", variant="secondary", visible=False)

                            with gr.Column(visible=False, elem_classes=["receipt-panel"]) as receipt_panel:
                                receipt_image = gr.Image(label="Receipt", visible=False, type="filepath")
                                receipt_text  = gr.Textbox(label="Receipt content", visible=False, lines=12, interactive=False)

                            gr.Markdown("---")
                            agent_rec_md    = gr.Markdown("")
                            gr.Markdown("---")
                            claim_details_md = gr.Markdown("")
                            gr.Markdown("---")
                            gate_analysis_md = gr.Markdown("")

                        # ── RIGHT COLUMN ──────────────────────────────
                        # 1. Ask the agent (accordion + chatbot)
                        # 2. Examiner decision block
                        with gr.Column(scale=2):

                            with gr.Accordion("💬 Ask the agent", open=True):
                                qa_chatbot = gr.Chatbot(
                                    label="",
                                    height=300,
                                    # bubble_full_width=False,
                                    show_label=False,
                                    # type="messages",
                                )
                                qa_input = gr.Textbox(
                                    placeholder=(
                                        "Ask about this claim, the rules, "
                                        "or why the confidence score is what it is..."
                                    ),
                                    lines=2,
                                    show_label=False,
                                )
                                ask_btn = gr.Button("Ask →", variant="secondary", size="sm")

                            gr.Markdown("---")
                            gr.Markdown("### Examiner decision")

                            examiner_notes   = gr.Textbox(
                                label="Notes *",
                                placeholder="Required — brief reason for this decision",
                                lines=2,
                                info="Mandatory for all decisions. Forms the audit trail.",
                            )
                            override_reason  = gr.Textbox(
                                label="Override reason *",
                                placeholder="Required if your decision differs from the agent recommendation",
                                lines=2,
                                info="Mandatory when overriding the agent. Leave blank if you agree.",
                            )
                            agent_hint_md    = gr.Markdown("*Load a claim to see the agent recommendation.*")

                            with gr.Row():
                                approve_btn  = gr.Button("✅ Approve", variant="primary")
                                reject_btn   = gr.Button("❌ Reject",  variant="stop")
                            with gr.Row():
                                pend_btn     = gr.Button("⏸ Pend",     variant="secondary")
                                escalate_btn = gr.Button("⚠️ Escalate", variant="secondary")
                            route_btn        = gr.Button("🔵 Route to HCSA", variant="secondary")

                            decision_status  = gr.Markdown("")

                            gr.Markdown("---")
                            gr.Markdown("### 🧠 Feedback for agent learning *(optional)*")
                            feedback_note    = gr.Textbox(
                                placeholder="What did the agent get right or wrong? This feeds the learning loop.",
                                lines=2,
                                show_label=False,
                            )

                # ── SHARED STATE ───────────────────────────────────────
                current_claim_id    = gr.State("")
                current_agent_dec   = gr.State("")
                # Stores the current filter so refresh works after returning from detail
                current_filter      = gr.State("All")

                # ── HELPER: full detail outputs list ──────────────────
                def _all_detail_outputs():
                    return [
                        detail_header,
                        agent_rec_md,
                        claim_details_md,
                        gate_analysis_md,
                        agent_hint_md,
                        current_claim_id,
                        current_agent_dec,
                        qa_chatbot,
                        decision_status,
                        # Receipt panel reset
                        receipt_panel,
                        receipt_image,
                        receipt_text,
                        close_receipt_btn,
                        view_receipt_btn,
                        # Text boxes — cleared fresh on every claim load
                        examiner_notes,
                        override_reason,
                        feedback_note,
                    ]

                def _render_detail(cid: str):
                    """Load and render all detail view components for a claim."""
                    claim = get_claim(cid)
                    if not claim:
                        return (
                            f"Claim #{cid} not found.",
                            "", "", "",
                            "*Claim not found.*",
                            cid, "",
                            [],
                            "",
                            gr.update(visible=False),
                            gr.update(visible=False, value=None),
                            gr.update(visible=False, value=""),
                            gr.update(visible=False),
                            gr.update(visible=True),
                        )

                    adj = claim.get("adjudicator_output") or {}
                    agent_dec = adj.get("decision") or ""
                    dec_icon  = _decision_icon(agent_dec)
                    code      = adj.get("reason_code") or "—"

                    # Check if claim already decided — show lock message
                    status = claim.get("status", "awaiting_review")
                    if status != "awaiting_review":
                        hint = (
                            f"🔒 *This claim has already been decided: "
                            f"**{status.upper()}**. No further action can be taken.*"
                        )
                    else:
                        hint = (
                            f"*Agent recommends **{dec_icon} {agent_dec}** ({code}). "
                            f"Override reason required only if you choose a different action.*"
                        )

                    return (
                        fmt_claim_header(claim),
                        fmt_agent_recommendation(claim),
                        fmt_claim_details(claim),
                        fmt_gate_analysis(claim),
                        hint,
                        cid,
                        agent_dec,
                        [],          # reset chat
                        "",          # clear decision status
                        gr.update(visible=False),        # receipt panel hidden
                        gr.update(visible=False, value=None),
                        gr.update(visible=False, value=""),
                        gr.update(visible=False),        # close btn hidden
                        gr.update(visible=True),         # view btn visible
                        gr.update(value=""),             # examiner_notes — cleared
                        gr.update(value=""),             # override_reason — cleared
                        gr.update(value=""),             # feedback_note  — cleared
                    )

                # ── EVENT: view claim button ───────────────────────────
                def on_view_claim(claim_id_raw: str):
                    cid = (claim_id_raw or "").strip().zfill(3)
                    if not cid.strip("0"):
                        return (
                            gr.update(),            # queue stays as-is
                            gr.update(),            # detail stays as-is
                            "⚠️ Please enter a claim ID.",
                            *[gr.update()] * len(_all_detail_outputs()),
                        )
                    claim = get_claim(cid)
                    if not claim:
                        return (
                            gr.update(),
                            gr.update(),
                            f"⚠️ Claim {cid} not found. Check the ID and try again.",
                            *[gr.update()] * len(_all_detail_outputs()),
                        )
                    detail_vals = _render_detail(cid)
                    return (
                        gr.update(visible=False),   # hide queue_view
                        gr.update(visible=True),    # show detail_view
                        "",                         # clear error
                        *detail_vals,
                    )

                view_claim_outputs = [
                    queue_view, detail_view, open_error_md
                ] + _all_detail_outputs()

                view_claim_btn.click(
                    on_view_claim,
                    inputs=[claim_id_input],
                    outputs=view_claim_outputs,
                )
                claim_id_input.submit(
                    on_view_claim,
                    inputs=[claim_id_input],
                    outputs=view_claim_outputs,
                )

                # ── EVENT: back button ─────────────────────────────────
                def go_back(filter_val):
                    return (
                        gr.update(visible=True),    # show queue
                        gr.update(visible=False),   # hide detail
                        build_claims_table(filter_val),
                        _fmt_stats(get_stats()),
                        "",   # clear decision status when returning
                    )

                back_btn.click(
                    go_back,
                    inputs=[current_filter],
                    outputs=[queue_view, detail_view, claims_table, stats_md, decision_status],
                )

                # ── EVENT: filter + refresh ────────────────────────────
                def refresh_queue(filter_val):
                    return (
                        build_claims_table(filter_val),
                        _fmt_stats(get_stats()),
                        filter_val,
                    )

                refresh_btn.click(
                    refresh_queue,
                    inputs=[filter_radio],
                    outputs=[claims_table, stats_md, current_filter],
                )
                filter_radio.change(
                    refresh_queue,
                    inputs=[filter_radio],
                    outputs=[claims_table, stats_md, current_filter],
                )

                # ── EVENT: ask the agent ───────────────────────────────
                def handle_ask(question, history, cid):
                    if not cid:
                        history = list(history or [])
                        history.append({"role": "assistant", "content": "⚠️ No claim loaded."})
                        return history, ""
                    if not (question or "").strip():
                        return list(history or []), ""

                    history = list(history or [])
                    history.append({"role": "user", "content": question.strip()})
                    try:
                        answer = ask_question(cid, question.strip())
                    except Exception as e:
                        answer = f"❌ Could not get an answer: {e}"
                    history.append({"role": "assistant", "content": answer})
                    return history, ""

                ask_btn.click(
                    handle_ask,
                    inputs=[qa_input, qa_chatbot, current_claim_id],
                    outputs=[qa_chatbot, qa_input],
                )
                qa_input.submit(
                    handle_ask,
                    inputs=[qa_input, qa_chatbot, current_claim_id],
                    outputs=[qa_chatbot, qa_input],
                )

                # ── EVENT: examiner decisions ──────────────────────────
                _DECISION_MAP = {
                    "approved":  "APPROVE",
                    "rejected":  "DENY",
                    "pended":    "PEND",
                    "escalated": "ESCALATE",
                    "routed":    "ROUTE_HCSA",
                }

                def handle_decision(label, cid, notes, override, feedback, agent_dec):
                    if not cid:
                        return "", fmt_claim_header(None)

                    # Mandatory notes
                    if not (notes or "").strip():
                        return (
                            "⚠️ **Notes are required.** Add a brief reason before submitting.",
                            fmt_claim_header(get_claim(cid)),
                        )

                    # Mandatory override reason when overriding
                    agent_as_label = _DECISION_MAP.get(label, "")
                    is_override = bool(agent_dec and agent_as_label and agent_as_label != agent_dec)
                    if is_override and not (override or "").strip():
                        return (
                            f"⚠️ **Override reason required.** "
                            f"Agent recommended **{agent_dec}** — you selected **{label.upper()}**. "
                            f"Please explain why.",
                            fmt_claim_header(get_claim(cid)),
                        )

                    result = submit_decision(
                        claim_id        = cid,
                        decision        = label,
                        notes           = notes.strip(),
                        override_reason = (override or "").strip(),
                        feedback_note   = (feedback or "").strip(),
                    )

                    # Check if already decided
                    if result.get("_already_decided"):
                        return (
                            f"🔒 {result.get('error', 'Already decided.')}",
                            fmt_claim_header(get_claim(cid)),
                        )

                    if result.get("error"):
                        return f"❌ {result['error']}", fmt_claim_header(get_claim(cid))

                    override_note = " *(override)*" if is_override else ""
                    status_msg = f"✅ Claim #{cid} → **{label.upper()}**{override_note}"
                    return status_msg, fmt_claim_header(result)

                decision_inputs = [
                    current_claim_id,
                    examiner_notes,
                    override_reason,
                    feedback_note,
                    current_agent_dec,
                ]
                decision_outputs = [decision_status, detail_header]

                for btn, label in [
                    (approve_btn,  "approved"),
                    (reject_btn,   "rejected"),
                    (pend_btn,     "pended"),
                    (escalate_btn, "escalated"),
                    (route_btn,    "routed"),
                ]:
                    btn.click(
                        (lambda lbl: lambda *args: handle_decision(lbl, *args))(label),
                        inputs=decision_inputs,
                        outputs=decision_outputs,
                    )

                # ── EVENT: view receipt ────────────────────────────────
                def show_receipt(cid):
                    claim = get_claim(cid) if cid else None
                    img_path, text_content, ftype = load_receipt_for_display(claim)
                    if ftype == "image" and img_path:
                        return (
                            gr.update(visible=True),
                            gr.update(visible=True,  value=img_path),
                            gr.update(visible=False, value=""),
                            gr.update(visible=True),    # close btn
                            gr.update(visible=False),   # view btn
                        )
                    elif ftype == "text" and text_content:
                        return (
                            gr.update(visible=True),
                            gr.update(visible=False, value=None),
                            gr.update(visible=True,  value=text_content),
                            gr.update(visible=True),
                            gr.update(visible=False),
                        )
                    else:
                        return (
                            gr.update(visible=True),
                            gr.update(visible=False),
                            gr.update(visible=True, value="Receipt file not available."),
                            gr.update(visible=True),
                            gr.update(visible=False),
                        )

                receipt_outputs = [receipt_panel, receipt_image, receipt_text, close_receipt_btn, view_receipt_btn]

                view_receipt_btn.click(
                    show_receipt,
                    inputs=[current_claim_id],
                    outputs=receipt_outputs,
                )

                def hide_receipt():
                    return (
                        gr.update(visible=False),
                        gr.update(visible=False, value=None),
                        gr.update(visible=False, value=""),
                        gr.update(visible=False),
                        gr.update(visible=True),
                    )

                close_receipt_btn.click(
                    hide_receipt,
                    outputs=receipt_outputs,
                )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=True, debug=True, show_error=True)