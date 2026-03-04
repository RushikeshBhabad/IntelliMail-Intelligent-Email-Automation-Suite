"""
graph.py
────────
LangGraph workflows for the Email Automation system.

Three independent agents
────────────────────────
1. Outreach Agent   – HR bulk outreach (existing)
2. Reply Agent      – Read inbox → classify → generate reply
3. Compose Agent    – User idea → full professional email

Each agent is built as a separate StateGraph and compiled independently.
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from resume_parser import parse_resume
from research_tool import research_company as _research_company
from email_sender import (
    send_email as _send_email,
    send_reply as _send_reply,
    send_composed_email as _send_composed_email,
    log_sent_email,
    is_already_sent,
    rate_limited_sleep,
)


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 1 — OUTREACH AGENT STATE & NODES
# ═══════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    # Loaded once
    companies: list[dict]
    resume_summary: str
    resume_path: str

    # Per-company iteration
    current_index: int
    company_research: str
    generated_email: str
    email_subject: str

    # Human review
    approval_status: str          # "approve" | "regenerate" | "skip"
    edited_email: str
    edited_subject: str

    # SMTP config
    smtp_server: str
    smtp_port: int
    sender_email: str
    sender_password: str

    # API keys
    groq_api_key: str
    tavily_api_key: str

    # Logging
    sent_log: list[dict]
    project_dir: str


def load_data(state: AgentState) -> dict:
    """Read the Excel file and parse the resume PDF."""
    excel_path = state.get("excel_path", "")
    resume_path = state.get("resume_path", "")
    groq_api_key = state["groq_api_key"]

    df = pd.read_excel(excel_path)
    df.columns = [c.strip() for c in df.columns]
    companies = df.to_dict(orient="records")

    resume_data = parse_resume(resume_path, groq_api_key)

    return {
        "companies": companies,
        "resume_summary": resume_data["summary"],
        "current_index": 0,
        "sent_log": [],
    }


def research_company_node(state: AgentState) -> dict:
    """Look up the current company on the internet and summarise."""
    idx = state["current_index"]
    company = state["companies"][idx]

    company_name = company.get("Company Name", company.get("company_name", "Unknown"))
    industry = company.get("Industry", company.get("industry", ""))
    location = company.get("Location", company.get("location", ""))

    summary = _research_company(
        company_name=company_name,
        industry=industry,
        location=location,
        tavily_api_key=state["tavily_api_key"],
        groq_api_key=state["groq_api_key"],
    )

    return {"company_research": summary}


def generate_email_node(state: AgentState) -> dict:
    """Draft a personalised internship outreach email."""
    idx = state["current_index"]
    company = state["companies"][idx]

    company_name = company.get("Company Name", company.get("company_name", "Unknown"))

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=state["groq_api_key"],
        temperature=0.7,
        max_tokens=500,
    )

    system_prompt = (
        "You are a professional outreach assistant writing highly personalised "
        "internship emails in HTML format. Follow this EXACT paragraph structure:\n\n"
        "PARAGRAPH 1: Start by introducing yourself — mention your name (from resume), "
        "that you are a 3rd year undergraduate student from PICT, Pune. "
        "Then add exactly 1 sentence about why you are interested in THIS specific company "
        "(use company research to personalise).\n\n"
        "PARAGRAPH 2: Talk about your skills and projects. Mention your full-stack skills "
        "(React, Next.js, Node.js, etc.) AND your Database skills also mention your AI/ML skills. "
        "List 2-3 specific projects from the resume with brief descriptions. "
        "Mention you can help with both full-stack development and AI initiatives.\n\n"
        "PARAGRAPH 3: Short closing — mention that your resume is attached for reference, "
        "express enthusiasm for an opportunity to discuss further, and end with a professional sign-off.\n\n"
        "FORMATTING RULES:\n"
        "- Use <b>bold</b> HTML tags for important words: your name, company name, key skills, "
        "project names, role title, and college name.\n"
        "- Keep it 150-200 words.\n"
        "- Do NOT include a subject line.\n"
        "- Do NOT use placeholders like [Your Name] — use the actual name from the resume.\n"
        "- Professional but enthusiastic tone.\n"
        "- Best regards should be followed by your name and Best regards should be on a new line "
        "and name should be on a new line below best regards."
    )

    user_prompt = (
        f"Write an internship application email for a Full Stack Intern role "
        f"at {company_name}.\n\n"
        f"Company Research:\n{state['company_research']}\n\n"
        f"My Resume Summary:\n{state['resume_summary']}\n\n"
        "Follow the 3-paragraph structure and use <b> tags for important words."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # Extract candidate name for subject
    resume_lines = state["resume_summary"].split("\n")
    candidate_name = resume_lines[0] if resume_lines else "Candidate"
    for prefix in ["**Name:**", "Name:", "**", "##"]:
        candidate_name = candidate_name.replace(prefix, "")
    candidate_name = candidate_name.strip().strip("*").strip()
    if len(candidate_name) > 50:
        candidate_name = "Candidate"

    subject = f"Application for Full Stack Intern – {candidate_name}"

    return {
        "generated_email": response.content.strip(),
        "email_subject": subject,
        "approval_status": "",
        "edited_email": "",
        "edited_subject": "",
    }


def human_review_node(state: AgentState) -> dict:
    """Interrupt point — Streamlit handles approval."""
    return {}


def send_email_node(state: AgentState) -> dict:
    """Send the approved email and log the result."""
    idx = state["current_index"]
    company = state["companies"][idx]

    company_name = company.get("Company Name", company.get("company_name", "Unknown"))
    to_email = company.get("Email ID", company.get("email_id", company.get("Email", "")))

    body = state.get("edited_email") or state["generated_email"]
    subject = state.get("edited_subject") or state["email_subject"]
    project_dir = state.get("project_dir", ".")

    result = _send_email(
        to_email=to_email,
        subject=subject,
        body=body,
        attachment_path=state.get("resume_path"),
        smtp_server=state["smtp_server"],
        smtp_port=state["smtp_port"],
        sender_email=state["sender_email"],
        sender_password=state["sender_password"],
    )

    log_sent_email(
        company=company_name,
        recipient_email=to_email,
        subject=subject,
        success=result["success"],
        error=result.get("error"),
        log_dir=project_dir,
    )

    log_entry = {
        "company": company_name,
        "email": to_email,
        "status": "SENT" if result["success"] else "FAILED",
        "error": result.get("error", ""),
    }
    sent_log = list(state.get("sent_log", []))
    sent_log.append(log_entry)

    rate_limited_sleep(2.0)
    return {"sent_log": sent_log}


def route_after_review(state: AgentState) -> str:
    status = state.get("approval_status", "")
    if status == "approve":
        return "send_email"
    elif status == "regenerate":
        return "generate_email"
    else:
        return END


def build_graph():
    """Construct and compile the Outreach Agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("load_data", load_data)
    graph.add_node("research_company", research_company_node)
    graph.add_node("generate_email", generate_email_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("send_email", send_email_node)

    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "research_company")
    graph.add_edge("research_company", "generate_email")
    graph.add_edge("generate_email", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "send_email": "send_email",
            "generate_email": "generate_email",
            END: END,
        },
    )
    graph.add_edge("send_email", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )
    return compiled, checkpointer


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 2 — INBOX REPLY AGENT STATE & NODES
# ═══════════════════════════════════════════════════════════════════════════

class ReplyAgentState(TypedDict, total=False):
    # Inbox data
    fetched_emails: list[dict]
    selected_email: dict

    # Classification
    classification: dict          # {"intent": str, "requires_reply": bool}

    # Generated reply
    reply_subject: str
    reply_body: str

    # Human review
    approval_status: str          # "send" | "regenerate" | "discard"
    edited_reply: str
    edited_subject: str

    # User description for tone/context
    user_description: str

    # SMTP / IMAP config
    smtp_server: str
    smtp_port: int
    imap_server: str
    imap_port: int
    sender_email: str
    sender_password: str

    # API
    groq_api_key: str


def fetch_emails_node(state: ReplyAgentState) -> dict:
    """Fetch emails from IMAP inbox."""
    from email_fetcher import fetch_emails

    emails = fetch_emails(
        imap_server=state.get("imap_server", "imap.gmail.com"),
        email_address=state["sender_email"],
        email_password=state["sender_password"],
        imap_port=state.get("imap_port", 993),
        filter_mode="unread",
        max_results=20,
    )
    return {"fetched_emails": emails}


def classify_email_node(state: ReplyAgentState) -> dict:
    """LLM classifies the selected email by intent."""
    email_data = state["selected_email"]

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=state["groq_api_key"],
        temperature=0,
        max_tokens=200,
    )

    system_prompt = (
        "You are an email classifier. Given an email, classify its intent "
        "into exactly ONE of these categories:\n"
        "- job_reply (response to a job application)\n"
        "- meeting (meeting request or scheduling)\n"
        "- recruiter (recruiter reaching out)\n"
        "- question (someone asking a question)\n"
        "- newsletter (newsletter or promotional)\n"
        "- spam (spam or irrelevant)\n"
        "- other (anything else)\n\n"
        "Also decide if the email requires a reply (true/false).\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"intent": "<category>", "requires_reply": true/false}'
    )

    user_prompt = (
        f"From: {email_data.get('sender', '')} <{email_data.get('sender_email', '')}>\n"
        f"Subject: {email_data.get('subject', '')}\n\n"
        f"Body:\n{email_data.get('body', '')[:2000]}"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    import json
    try:
        text = response.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            classification = json.loads(text[start:end])
        else:
            classification = {"intent": "other", "requires_reply": True}
    except json.JSONDecodeError:
        classification = {"intent": "other", "requires_reply": True}

    return {"classification": classification}


def generate_reply_node(state: ReplyAgentState) -> dict:
    """LLM generates a professional reply to the selected email."""
    email_data = state["selected_email"]
    classification = state.get("classification", {})
    user_desc = state.get("user_description", "")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=state["groq_api_key"],
        temperature=0.7,
        max_tokens=500,
    )

    system_prompt = (
        "You are a professional email reply assistant. Write a reply to the "
        "email provided. The reply should be:\n"
        "- Professional and courteous\n"
        "- Concise (50-150 words)\n"
        "- Relevant to the email content and intent\n"
        "- In HTML format using <b> for emphasis where appropriate\n\n"
        "Do NOT include the subject line in the body.\n"
        "Do NOT use placeholders — write the actual reply.\n"
        "End with a professional sign-off."
    )

    user_prompt = (
        f"Email intent: {classification.get('intent', 'unknown')}\n\n"
        f"Original email:\n"
        f"From: {email_data.get('sender', '')} <{email_data.get('sender_email', '')}>\n"
        f"Subject: {email_data.get('subject', '')}\n"
        f"Body:\n{email_data.get('body', '')[:2000]}\n\n"
    )

    if user_desc:
        user_prompt += f"User instruction for tone/context:\n{user_desc}\n\n"

    user_prompt += "Write a professional reply to this email."

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    original_subject = email_data.get("subject", "")
    if original_subject.lower().startswith("re:"):
        reply_subject = original_subject
    else:
        reply_subject = f"Re: {original_subject}"

    return {
        "reply_body": response.content.strip(),
        "reply_subject": reply_subject,
        "approval_status": "",
        "edited_reply": "",
        "edited_subject": "",
    }


def reply_human_review_node(state: ReplyAgentState) -> dict:
    """Interrupt point for reply agent — Streamlit handles."""
    return {}


def send_reply_node(state: ReplyAgentState) -> dict:
    """Send the approved reply, preserving the email thread."""
    email_data = state["selected_email"]

    body = state.get("edited_reply") or state["reply_body"]
    subject = state.get("edited_subject") or state["reply_subject"]

    result = _send_reply(
        to_email=email_data["sender_email"],
        subject=subject,
        body=body,
        message_id=email_data.get("message_id", ""),
        in_reply_to=email_data.get("message_id", ""),
        smtp_server=state["smtp_server"],
        smtp_port=state["smtp_port"],
        sender_email=state["sender_email"],
        sender_password=state["sender_password"],
    )

    return {"send_result": result}


def route_after_reply_review(state: ReplyAgentState) -> str:
    status = state.get("approval_status", "")
    if status == "send":
        return "send_reply"
    elif status == "regenerate":
        return "generate_reply"
    else:
        return END


def build_reply_graph():
    """Construct and compile the Reply Agent graph."""
    graph = StateGraph(ReplyAgentState)

    graph.add_node("fetch_emails", fetch_emails_node)
    graph.add_node("classify_email", classify_email_node)
    graph.add_node("generate_reply", generate_reply_node)
    graph.add_node("human_review", reply_human_review_node)
    graph.add_node("send_reply", send_reply_node)

    graph.set_entry_point("fetch_emails")
    graph.add_edge("fetch_emails", "classify_email")
    graph.add_edge("classify_email", "generate_reply")
    graph.add_edge("generate_reply", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_after_reply_review,
        {
            "send_reply": "send_reply",
            "generate_reply": "generate_reply",
            END: END,
        },
    )
    graph.add_edge("send_reply", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )
    return compiled, checkpointer


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 3 — COMPOSE AGENT STATE & NODES
# ═══════════════════════════════════════════════════════════════════════════

class ComposeAgentState(TypedDict, total=False):
    # User input
    recipient_email: str
    brief_topic: str
    tone: str                     # "professional" | "friendly" | "formal" | "casual"
    attachment_paths: list[str]
    cc: str
    bcc: str

    # Generated email
    composed_subject: str
    composed_body: str

    # Human review
    approval_status: str          # "send" | "regenerate" | "discard"
    edited_body: str
    edited_subject: str

    # SMTP config
    smtp_server: str
    smtp_port: int
    sender_email: str
    sender_password: str

    # API
    groq_api_key: str


def compose_generate_node(state: ComposeAgentState) -> dict:
    """LLM creates a full email from a brief topic and tone."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=state["groq_api_key"],
        temperature=0.7,
        max_tokens=600,
    )

    tone = state.get("tone", "professional") or "professional"

    system_prompt = (
        "You are a professional email writing assistant. "
        "Given a brief topic and a desired tone, write a complete email.\n\n"
        "Rules:\n"
        "- Write ONLY the email body (no subject line in the body)\n"
        "- Use HTML format with <b> tags for emphasis\n"
        "- Match the requested tone exactly\n"
        "- Be concise but thorough (80-200 words)\n"
        "- Do NOT use placeholders like [Your Name] or [Company]\n"
        "- End with an appropriate sign-off\n\n"
        "Also generate a concise, appropriate subject line.\n\n"
        "Return your response in this exact format:\n"
        "SUBJECT: <subject line here>\n"
        "BODY:\n<email body here>"
    )

    user_prompt = (
        f"Write a {tone} email.\n\n"
        f"Topic / Idea:\n{state['brief_topic']}\n\n"
        f"Recipient: {state.get('recipient_email', 'the recipient')}\n\n"
        f"Tone: {tone}\n\n"
        "Return SUBJECT: and BODY: as specified."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    text = response.content.strip()

    # Parse SUBJECT and BODY
    subject = ""
    body = text
    if "SUBJECT:" in text and "BODY:" in text:
        parts = text.split("BODY:", 1)
        subject_part = parts[0]
        body = parts[1].strip() if len(parts) > 1 else text

        subj_lines = subject_part.split("SUBJECT:", 1)
        if len(subj_lines) > 1:
            subject = subj_lines[1].strip().split("\n")[0].strip()
    elif "SUBJECT:" in text:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("SUBJECT:"):
                subject = line.replace("SUBJECT:", "").strip()
                body = "\n".join(lines[i + 1:]).strip()
                break

    if not subject:
        subject = f"Regarding: {state['brief_topic'][:50]}"

    return {
        "composed_subject": subject,
        "composed_body": body,
        "approval_status": "",
        "edited_body": "",
        "edited_subject": "",
    }


def compose_human_review_node(state: ComposeAgentState) -> dict:
    """Interrupt point for compose agent — Streamlit handles."""
    return {}


def compose_send_node(state: ComposeAgentState) -> dict:
    """Send the composed email."""
    body = state.get("edited_body") or state["composed_body"]
    subject = state.get("edited_subject") or state["composed_subject"]

    result = _send_composed_email(
        to_email=state["recipient_email"],
        subject=subject,
        body=body,
        smtp_server=state["smtp_server"],
        smtp_port=state["smtp_port"],
        sender_email=state["sender_email"],
        sender_password=state["sender_password"],
        cc=state.get("cc", ""),
        bcc=state.get("bcc", ""),
        attachment_paths=state.get("attachment_paths"),
    )

    return {"send_result": result}


def route_after_compose_review(state: ComposeAgentState) -> str:
    status = state.get("approval_status", "")
    if status == "send":
        return "send_email"
    elif status == "regenerate":
        return "generate_email"
    else:
        return END


def build_compose_graph():
    """Construct and compile the Compose Agent graph."""
    graph = StateGraph(ComposeAgentState)

    graph.add_node("generate_email", compose_generate_node)
    graph.add_node("human_review", compose_human_review_node)
    graph.add_node("send_email", compose_send_node)

    graph.set_entry_point("generate_email")
    graph.add_edge("generate_email", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_after_compose_review,
        {
            "send_email": "send_email",
            "generate_email": "generate_email",
            END: END,
        },
    )
    graph.add_edge("send_email", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )
    return compiled, checkpointer
