"""
app.py - Streamlit UI for Email Automation Agent
Three tabs: HR Outreach | Inbox Reply | Compose Email
Run with: streamlit run app.py
"""

import os, re, sys, tempfile, uuid
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from graph import (
    build_graph, build_reply_graph, build_compose_graph,
    AgentState, ReplyAgentState, ComposeAgentState,
)
from email_sender import is_already_sent

load_dotenv(os.path.join(PROJECT_DIR, ".env"))

st.set_page_config(page_title="📧 Email Automation AI", page_icon="📧", layout="wide")

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*='st-'] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.main-header h1 { color: #e94560; font-size: 2.2rem; font-weight: 700; margin: 0; }
.main-header p  { color: #a8b2d1; font-size: 1rem; margin: 0.4rem 0 0; }
.info-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
}
.info-card h4 { color: #e94560; margin: 0 0 0.4rem; }
.info-card p  { color: #94a3b8; margin: 0; font-size: 0.9rem; }
.stButton > button { border-radius: 8px; font-weight: 600; transition: all 0.2s ease; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(233,69,96,0.3); }
section[data-testid='stSidebar'] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
section[data-testid='stSidebar'] .stMarkdown h2,
section[data-testid='stSidebar'] .stMarkdown h3 { color: #e94560; }
hr { border-color: #334155; }
.main .block-container { overflow: visible !important; }
.element-container { position: static !important; overflow: visible !important; }
[data-testid='stExpander'] { position: static !important; z-index: auto !important; margin-bottom: 1.5rem !important; }
.stTextInput, .stTextArea { position: static !important; z-index: auto !important; }
.research-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 1.5rem; font-size: 0.9rem;
    color: #cbd5e1; line-height: 1.6;
}
.research-card h4 { color: #e94560; margin: 0 0 0.5rem; }
.email-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem; transition: border-color 0.2s;
}
.email-card:hover { border-color: #e94560; }
.email-card h5 { color: #e2e8f0; margin: 0 0 0.3rem; font-size: 0.95rem; }
.email-card .sender { color: #e94560; font-weight: 600; font-size: 0.85rem; }
.email-card .preview { color: #94a3b8; font-size: 0.82rem; margin-top: 0.3rem; }
.email-card .timestamp { color: #64748b; font-size: 0.75rem; float: right; }
.intent-badge {
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; margin-left: 0.5rem;
}
.intent-job_reply  { background: #065f46; color: #6ee7b7; }
.intent-meeting    { background: #1e3a5f; color: #7dd3fc; }
.intent-recruiter  { background: #581c87; color: #d8b4fe; }
.intent-question   { background: #713f12; color: #fde68a; }
.intent-newsletter { background: #374151; color: #9ca3af; }
.intent-spam       { background: #7f1d1d; color: #fca5a5; }
.intent-other      { background: #334155; color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)


# SESSION STATE
def init_session():
    defaults = {
        "active_tab": "HR Outreach",
        "graph": None, "checkpointer": None,
        "thread_id": str(uuid.uuid4()),
        "companies": [], "resume_summary": "",
        "resume_path_tmp": "", "excel_path_tmp": "",
        "data_loaded": False, "current_company_idx": 0,
        "generated_email": "", "email_subject": "",
        "company_research": "", "email_generating": False,
        "email_ready": False, "sent_log": [],
        "inbox_emails": [], "inbox_loaded": False,
        "inbox_selected_idx": None, "inbox_classification": None,
        "inbox_reply_body": "", "inbox_reply_subject": "",
        "inbox_reply_ready": False, "inbox_generating": False,
        "compose_subject": "", "compose_body": "",
        "compose_ready": False, "compose_generating": False,
        "compose_attachments": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# HEADER
st.markdown("""
<div class="main-header">
    <h1>📧 Email Automation AI</h1>
    <p>HR Outreach • Inbox Reply • Smart Compose — powered by LangGraph + Groq (free)</p>
</div>
""", unsafe_allow_html=True)


# SIDEBAR
with st.sidebar:
    st.markdown("## 📧 Email Automation AI")
    st.markdown("---")

    st.markdown("### 🔑 API Keys")
    groq_api_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    tavily_api_key = st.text_input("Tavily API Key (HR Outreach)", value=os.getenv("TAVILY_API_KEY", ""), type="password")

    st.markdown("---")
    st.markdown("### ✉️ Email Credentials")
    sender_email = st.text_input("Email Address", value=os.getenv("EMAIL_ADDRESS", ""))
    sender_password = st.text_input("Email App Password", value=os.getenv("EMAIL_PASSWORD", ""), type="password")
    smtp_server = st.text_input("SMTP Server", value=os.getenv("SMTP_SERVER", "smtp.gmail.com"))
    smtp_port = st.number_input("SMTP Port", value=int(os.getenv("SMTP_PORT", "587")), min_value=1, max_value=65535)
    imap_server = st.text_input("IMAP Server", value=os.getenv("IMAP_SERVER", "imap.gmail.com"))

    st.markdown("---")
    st.markdown(
        '<div class="info-card"><h4>💡 Tip</h4>'
        '<p>For Gmail use an <b>App Password</b> (not your regular password). Enable 2-FA first.</p></div>',
        unsafe_allow_html=True,
    )


# HELPER
def save_uploaded_file(uploaded_file, suffix: str) -> str:
    tmp_dir = tempfile.mkdtemp()
    dest = os.path.join(tmp_dir, uploaded_file.name)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


# MAIN TABS
tab_outreach, tab_inbox, tab_compose = st.tabs([
    "1️⃣ HR Outreach",
    "2️⃣ Inbox Reply",
    "3️⃣ Compose Email",
])


# TAB 1 — HR OUTREACH
with tab_outreach:
    st.markdown("## 📤 HR Outreach Agent")
    st.caption("Upload resume + Excel → AI researches each company → generates personalised emails")

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        resume_file = st.file_uploader("Resume PDF", type=["pdf"], key="resume_uploader")
    with col_up2:
        excel_file = st.file_uploader("HR Contacts Excel", type=["xlsx", "xls"], key="excel_uploader")

    st.markdown("### 1️⃣ Load Data")
    col_l1, col_l2 = st.columns([3, 1])
    with col_l1:
        if resume_file and excel_file and groq_api_key:
            st.success("✅ Resume and Excel uploaded. API key provided.")
        else:
            missing = []
            if not resume_file:  missing.append("Resume PDF")
            if not excel_file:   missing.append("Excel file")
            if not groq_api_key: missing.append("Groq API Key")
            st.warning(f"⚠️ Missing: {', '.join(missing)}")

    with col_l2:
        load_btn = st.button("🚀 Load Data", disabled=not (resume_file and excel_file and groq_api_key), use_container_width=True)

    if load_btn:
        with st.spinner("📖 Parsing resume & reading Excel …"):
            resume_tmp = save_uploaded_file(resume_file, ".pdf")
            excel_tmp = save_uploaded_file(excel_file, ".xlsx")
            st.session_state["resume_path_tmp"] = resume_tmp
            st.session_state["excel_path_tmp"] = excel_tmp

            compiled_graph, checkpointer = build_graph()
            st.session_state["graph"] = compiled_graph
            st.session_state["checkpointer"] = checkpointer
            st.session_state["thread_id"] = str(uuid.uuid4())

            from resume_parser import parse_resume
            resume_data = parse_resume(resume_tmp, groq_api_key)
            df = pd.read_excel(excel_tmp)
            df.columns = [c.strip() for c in df.columns]
            companies = df.to_dict(orient="records")

            st.session_state["companies"] = companies
            st.session_state["resume_summary"] = resume_data["summary"]
            st.session_state["data_loaded"] = True
            st.session_state["sent_log"] = []
            st.session_state["current_company_idx"] = 0
            st.session_state["generated_email"] = ""
            st.session_state["email_subject"] = ""
            st.session_state["company_research"] = ""
            st.session_state["email_ready"] = False

        st.success(f"✅ Loaded **{len(companies)}** companies & parsed resume.")
        st.rerun()

    if st.session_state["data_loaded"] and st.session_state["resume_summary"]:
        with st.expander("📝 Resume Summary", expanded=False):
            st.markdown(st.session_state["resume_summary"])

    if st.session_state["data_loaded"]:
        st.markdown("---")
        st.markdown("### 2️⃣ Select Company & Generate Email")

        companies = st.session_state["companies"]

        def company_label(c):
            name = c.get("Company Name", c.get("company_name", "Unknown"))
            email = c.get("Email ID", c.get("email_id", c.get("Email", "")))
            return f"{name}  •  {email}"

        labels = [company_label(c) for c in companies]
        selected_label = st.selectbox("Choose a company", labels, index=st.session_state["current_company_idx"])
        selected_idx = labels.index(selected_label)
        st.session_state["current_company_idx"] = selected_idx
        selected_company = companies[selected_idx]

        cname = selected_company.get("Company Name", selected_company.get("company_name", ""))
        cindustry = selected_company.get("Industry", selected_company.get("industry", ""))
        clocation = selected_company.get("Location", selected_company.get("location", ""))
        cemail = selected_company.get("Email ID", selected_company.get("email_id", selected_company.get("Email", "")))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏢 Company", cname)
        c2.metric("🏭 Industry", cindustry or "—")
        c3.metric("📍 Location", clocation or "—")
        c4.metric("📧 Email", cemail or "—")

        already_sent = is_already_sent(cemail, log_dir=PROJECT_DIR)
        if already_sent:
            st.info("ℹ️ An email has already been sent to this address.")

        gen_btn = st.button("🤖 Research & Generate Email", use_container_width=True,
                            disabled=st.session_state.get("email_generating", False))

        if gen_btn:
            st.session_state["email_generating"] = True
            st.session_state["email_ready"] = False

            with st.status("Working …", expanded=True) as status:
                st.write("🔍 Researching company …")
                from research_tool import research_company as do_research
                company_research = do_research(
                    company_name=cname, industry=cindustry, location=clocation,
                    tavily_api_key=tavily_api_key, groq_api_key=groq_api_key,
                )
                st.session_state["company_research"] = company_research
                st.write("✅ Research complete.")

                st.write("✍️ Drafting email …")
                from langchain_groq import ChatGroq
                from langchain_core.messages import SystemMessage, HumanMessage

                llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key,
                               temperature=0.7, max_tokens=500)
                sys_p = (
                    "You are a professional outreach assistant writing highly personalised "
                    "internship emails in HTML format. Follow this EXACT paragraph structure:\n\n"
                    "PARAGRAPH 1: Introduce yourself \u2014 name (from resume), 3rd year undergrad from PICT, Pune. "
                    "1 sentence about why THIS company interests you.\n\n"
                    "PARAGRAPH 2: Skills and projects. Full-stack (React, Node.js, etc.) AND AI/ML. "
                    "List 2-3 projects. Mention you can help with full-stack dev and AI.\n\n"
                    "PARAGRAPH 3: Short closing \u2014 resume attached, enthusiasm, professional sign-off.\n\n"
                    "FORMATTING:\n- <b> for important words\n- 150-200 words\n- No subject line\n"
                    "- No placeholders\n- Professional but enthusiastic\n"
                    "- Best regards on new line, name below it."
                )
                usr_p = (
                    f"Write an internship application email for Full Stack Intern at {cname}.\n\n"
                    f"Company Research:\n{company_research}\n\n"
                    f"Resume Summary:\n{st.session_state['resume_summary']}\n\n" 
                    "Follow the 3-paragraph structure. Use <b> tags."
                )
                response = llm.invoke([SystemMessage(content=sys_p), HumanMessage(content=usr_p)])
                generated = response.content.strip()

                resume_lines = st.session_state["resume_summary"].split("\n")
                cand = resume_lines[0] if resume_lines else "Candidate"
                for pfx in ["**Name:**", "Name:", "**", "##"]:
                    cand = cand.replace(pfx, "")
                cand = cand.strip().strip("*").strip()
                if len(cand) > 50:
                    cand = "Candidate"
                subject = f"Application for Full Stack Intern \u2013 {cand}"

                st.session_state["generated_email"] = generated
                st.session_state["email_subject"] = subject
                st.session_state["email_ready"] = True
                st.session_state["email_generating"] = False
                status.update(label="✅ Email generated!", state="complete")
            st.rerun()

    # Review & Approve
    if st.session_state.get("email_ready"):
        st.markdown("---")
        st.markdown("### 3️⃣ Review & Approve Email")

        if st.session_state.get("company_research"):
            st.markdown(
                f'<div class="research-card"><h4>🔍 Company Research</h4>'
                f'{st.session_state["company_research"]}</div>',
                unsafe_allow_html=True,
            )

        edited_subject = st.text_input("📌 Subject Line", value=st.session_state["email_subject"], key="outreach_subj")
        edited_email = st.text_area("📝 Email Body", value=st.session_state["generated_email"], height=300, key="outreach_body")

        plain = re.sub(r"<[^>]+>", "", edited_email)
        wc = len(plain.split())
        marker = "✅" if 150 <= wc <= 200 else "⚠️ (target: 150–200)"
        st.caption(f"Word count: **{wc}** {marker}")

        st.markdown("#### 👁️ Preview")
        preview = edited_email.replace("\n", "<br>")
        st.markdown(
            f'<div style="background:#f8f9fa;color:#1a1a2e;padding:1.2rem 1.5rem;'
            f'border-radius:10px;font-family:Arial;font-size:14px;line-height:1.7;'
            f'border:1px solid #dee2e6;">{preview}</div>',
            unsafe_allow_html=True,
        )

        ca, cb, cc_ = st.columns(3)
        with ca:
            approve_btn = st.button("✅ Approve & Send", use_container_width=True, key="out_approve")
        with cb:
            regen_btn = st.button("🔄 Regenerate", use_container_width=True, key="out_regen")
        with cc_:
            skip_btn = st.button("⏭️ Skip", use_container_width=True, key="out_skip")

        if approve_btn:
            if not sender_email or not sender_password:
                st.error("❌ Enter email credentials in sidebar.")
            else:
                companies = st.session_state["companies"]
                idx = st.session_state["current_company_idx"]
                company = companies[idx]
                to_email = company.get("Email ID", company.get("email_id", company.get("Email", "")))
                cname_s = company.get("Company Name", company.get("company_name", "Unknown"))

                with st.spinner("📤 Sending …"):
                    from email_sender import send_email, log_sent_email, rate_limited_sleep
                    result = send_email(
                        to_email=to_email, subject=edited_subject, body=edited_email,
                        attachment_path=st.session_state.get("resume_path_tmp"),
                        smtp_server=smtp_server, smtp_port=int(smtp_port),
                        sender_email=sender_email, sender_password=sender_password,
                    )
                    log_sent_email(company=cname_s, recipient_email=to_email,
                                   subject=edited_subject, success=result["success"],
                                   error=result.get("error"), log_dir=PROJECT_DIR)
                    st.session_state["sent_log"].append({
                        "company": cname_s, "email": to_email,
                        "status": "SENT" if result["success"] else "FAILED",
                        "error": result.get("error", ""),
                    })

                if result["success"]:
                    st.success(f"✅ Sent to **{to_email}**!")
                    st.balloons()
                else:
                    st.error(f"❌ Failed: {result['error']}")
                st.session_state["email_ready"] = False
                st.session_state["generated_email"] = ""
                st.session_state["email_subject"] = ""
                rate_limited_sleep(1.0)

        if regen_btn:
            st.session_state["email_ready"] = False
            st.session_state["generated_email"] = ""
            st.session_state["email_subject"] = ""
            st.rerun()

        if skip_btn:
            st.session_state["email_ready"] = False
            st.session_state["generated_email"] = ""
            st.session_state["email_subject"] = ""
            st.info("⏭️ Skipped.")

    # Sent Log
    if st.session_state["sent_log"]:
        st.markdown("---")
        st.markdown("### 📊 Sent Email Log")
        st.dataframe(pd.DataFrame(st.session_state["sent_log"]), use_container_width=True)
        csv_path = os.path.join(PROJECT_DIR, "sent_emails_log.csv")
        if os.path.isfile(csv_path):
            with open(csv_path, "r") as f:
                st.download_button("📥 Download CSV Log", data=f.read(),
                                   file_name="sent_emails_log.csv", mime="text/csv")


# TAB 2 — INBOX REPLY
with tab_inbox:
    st.markdown("## 📩 AI Inbox Reply Agent")
    st.caption("Fetch unread emails → AI classifies & generates replies → you approve before sending")

    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        if sender_email and sender_password and groq_api_key:
            st.success("✅ Credentials ready.")
        else:
            miss = []
            if not sender_email:    miss.append("Email")
            if not sender_password: miss.append("Password")
            if not groq_api_key:    miss.append("Groq Key")
            st.warning(f"⚠️ Missing: {', '.join(miss)}")

    with col_f2:
        fetch_btn = st.button("📥 Fetch Unread Emails",
                              disabled=not (sender_email and sender_password),
                              use_container_width=True)

    if fetch_btn:
        with st.spinner("📬 Connecting to inbox via IMAP …"):
            try:
                from email_fetcher import fetch_emails
                emails = fetch_emails(
                    imap_server=imap_server,
                    email_address=sender_email,
                    email_password=sender_password,
                    filter_mode="unread",
                    max_results=20,
                )
                st.session_state["inbox_emails"] = emails
                st.session_state["inbox_loaded"] = True
                st.session_state["inbox_selected_idx"] = None
                st.session_state["inbox_classification"] = None
                st.session_state["inbox_reply_ready"] = False
                if emails:
                    st.success(f"✅ Fetched **{len(emails)}** unread emails.")
                else:
                    st.info("📭 No unread emails found.")
            except Exception as e:
                st.error(f"❌ IMAP error: {e}")

    # Email List
    if st.session_state["inbox_loaded"] and st.session_state["inbox_emails"]:
        st.markdown("---")
        st.markdown("### 📬 Unread Emails")

        emails = st.session_state["inbox_emails"]

        for i, em in enumerate(emails):
            body_prev = em["body"][:120].replace("\n", " ")
            if len(em["body"]) > 120:
                body_prev += "…"
            ts = em.get("timestamp", "")

            st.markdown(
                f'<div class="email-card">'
                f'<span class="timestamp">{ts}</span>'
                f'<span class="sender">📧 {em["sender"]} &lt;{em["sender_email"]}&gt;</span>'
                f'<h5>{em["subject"]}</h5>'
                f'<p class="preview">{body_prev}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if st.button("💬 Generate Reply", key=f"reply_btn_{i}", use_container_width=True):
                st.session_state["inbox_selected_idx"] = i
                st.session_state["inbox_reply_ready"] = False
                st.session_state["inbox_generating"] = True

                selected = emails[i]

                with st.status("🤖 Processing …", expanded=True) as status:
                    st.write("🏷️ Classifying email intent …")
                    from langchain_groq import ChatGroq as CG
                    from langchain_core.messages import SystemMessage as SM, HumanMessage as HM
                    import json

                    llm_c = CG(model="llama-3.3-70b-versatile", api_key=groq_api_key,
                               temperature=0, max_tokens=200)
                    cls_sys = (
                        "You are an email classifier. Classify intent into ONE of: "
                        "job_reply, meeting, recruiter, question, newsletter, spam, other.\n"
                        "Decide if it requires a reply (true/false).\n"
                        'Respond ONLY with JSON: {"intent": "<category>", "requires_reply": true/false}'
                    )
                    cls_usr = (
                        f"From: {selected['sender']} <{selected['sender_email']}>\n" 
                        f"Subject: {selected['subject']}\n\n" 
                        f"Body:\n{selected['body'][:2000]}" 
                    )
                    cls_resp = llm_c.invoke([SM(content=cls_sys), HM(content=cls_usr)])
                    try:
                        txt = cls_resp.content.strip()
                        s_idx, e_idx = txt.find("{"), txt.rfind("}") + 1
                        classification = json.loads(txt[s_idx:e_idx]) if s_idx >= 0 and e_idx > s_idx else {"intent": "other", "requires_reply": True}
                    except Exception:
                        classification = {"intent": "other", "requires_reply": True}
                    st.session_state["inbox_classification"] = classification
                    st.write(f"✅ Intent: **{classification['intent']}** | Reply needed: **{classification['requires_reply']}**")

                    st.write("✍️ Generating reply …")
                    llm_r = CG(model="llama-3.3-70b-versatile", api_key=groq_api_key,
                               temperature=0.7, max_tokens=500)
                    rep_sys = (
                        "You are a professional email reply assistant. Write a reply that is:\n"
                        "- Professional and courteous\n- Concise (50-150 words)\n"
                        "- In HTML format using <b> for emphasis\n"
                        "- No placeholders\n- Professional sign-off"
                    )
                    rep_usr = (
                        f"Email intent: {classification['intent']}\n\n" 
                        f"Original email:\nFrom: {selected['sender']} <{selected['sender_email']}>\n" 
                        f"Subject: {selected['subject']}\nBody:\n{selected['body'][:2000]}\n\n" 
                        "Write a professional reply."
                    )
                    rep_resp = llm_r.invoke([SM(content=rep_sys), HM(content=rep_usr)])

                    orig_subj = selected.get("subject", "")
                    reply_subj = orig_subj if orig_subj.lower().startswith("re:") else f"Re: {orig_subj}"

                    st.session_state["inbox_reply_body"] = rep_resp.content.strip()
                    st.session_state["inbox_reply_subject"] = reply_subj
                    st.session_state["inbox_reply_ready"] = True
                    st.session_state["inbox_generating"] = False
                    status.update(label="✅ Reply generated!", state="complete")

                st.rerun()

    # Reply Review
    if st.session_state.get("inbox_reply_ready") and st.session_state.get("inbox_selected_idx") is not None:
        st.markdown("---")
        st.markdown("### ✉️ Review & Send Reply")

        idx = st.session_state["inbox_selected_idx"]
        original = st.session_state["inbox_emails"][idx]
        classification = st.session_state.get("inbox_classification", {})

        intent = classification.get("intent", "other")
        st.markdown(
            f'<div class="research-card">'
            f'<h4>📨 Original Email '
            f'<span class="intent-badge intent-{intent}">{intent}</span></h4>'
            f'<b>From:</b> {original["sender"]} &lt;{original["sender_email"]}&gt;<br>'
            f'<b>Subject:</b> {original["subject"]}<br><br>'
            f'{original["body"][:1500]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        ed_subj = st.text_input("📌 Reply Subject", value=st.session_state["inbox_reply_subject"], key="reply_subj_ed")
        ed_body = st.text_area("📝 Reply Body", value=st.session_state["inbox_reply_body"], height=250, key="reply_body_ed")

        st.markdown("#### 👁️ Reply Preview")
        preview_r = ed_body.replace("\n", "<br>")
        st.markdown(
            f'<div style="background:#f8f9fa;color:#1a1a2e;padding:1rem 1.2rem;'
            f'border-radius:10px;font-family:Arial;font-size:14px;line-height:1.6;'
            f'border:1px solid #dee2e6;">{preview_r}</div>',
            unsafe_allow_html=True,
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            send_reply_btn = st.button("📤 Send Reply", use_container_width=True, key="send_reply")
        with r2:
            regen_reply = st.button("🔄 Regenerate", use_container_width=True, key="regen_reply")
        with r3:
            discard_reply = st.button("🗑️ Discard", use_container_width=True, key="discard_reply")

        if send_reply_btn:
            if not sender_email or not sender_password:
                st.error("❌ Enter email credentials in sidebar.")
            else:
                with st.spinner("📤 Sending reply …"):
                    from email_sender import send_reply
                    result = send_reply(
                        to_email=original["sender_email"],
                        subject=ed_subj,
                        body=ed_body,
                        message_id=original.get("message_id", ""),
                        in_reply_to=original.get("message_id", ""),
                        smtp_server=smtp_server,
                        smtp_port=int(smtp_port),
                        sender_email=sender_email,
                        sender_password=sender_password,
                    )
                if result["success"]:
                    st.success(f"✅ Reply sent to **{original['sender_email']}**!")
                    st.balloons()
                    try:
                        from email_fetcher import mark_as_read
                        mark_as_read(imap_server, sender_email, sender_password, original["uid"])
                    except Exception:
                        pass
                else:
                    st.error(f"❌ Failed: {result['error']}")
                st.session_state["inbox_reply_ready"] = False
                st.session_state["inbox_selected_idx"] = None

        if regen_reply:
            st.session_state["inbox_reply_ready"] = False
            st.session_state["inbox_reply_body"] = ""
            st.session_state["inbox_reply_subject"] = ""
            st.rerun()

        if discard_reply:
            st.session_state["inbox_reply_ready"] = False
            st.session_state["inbox_selected_idx"] = None
            st.info("🗑️ Reply discarded.")


# TAB 3 — COMPOSE EMAIL
with tab_compose:
    st.markdown("## ✍️ Smart Compose Email")
    st.caption("Write a short idea → AI generates a complete professional email")

    st.markdown("### 1️⃣ Email Details")

    comp_recipient = st.text_input("📧 Recipient Email", key="comp_recipient")
    comp_cc = st.text_input("📋 CC (comma-separated)", key="comp_cc")
    comp_bcc = st.text_input("📋 BCC (comma-separated)", key="comp_bcc")
    comp_topic = st.text_area("💡 Brief Topic / Idea", height=100, key="comp_topic",
                              placeholder="e.g. Follow up for internship position at Google")
    comp_tone = st.selectbox("🎨 Tone", ["Professional", "Friendly", "Formal", "Casual", "Persuasive"], key="comp_tone")
    comp_attachments = st.file_uploader("📎 Attachments", accept_multiple_files=True, key="comp_attach")

    can_generate = bool(comp_recipient and comp_topic and groq_api_key)
    gen_compose = st.button("🤖 Generate Email", use_container_width=True,
                            disabled=not can_generate, key="gen_compose_btn")

    if gen_compose:
        st.session_state["compose_ready"] = False
        st.session_state["compose_generating"] = True

        att_paths = []
        if comp_attachments:
            for att in comp_attachments:
                att_paths.append(save_uploaded_file(att, ""))
        st.session_state["compose_attachments"] = att_paths

        with st.status("✍️ Generating email …", expanded=True) as status:
            from langchain_groq import ChatGroq as CG3
            from langchain_core.messages import SystemMessage as SM3, HumanMessage as HM3

            llm_comp = CG3(model="llama-3.3-70b-versatile", api_key=groq_api_key,
                           temperature=0.7, max_tokens=600)
            tone_val = comp_tone.lower()
            sys_comp = (
                "You are a professional email writing assistant. "
                "Given a brief topic and tone, write a complete email.\n\n"
                "Rules:\n- Write ONLY the email body (no subject in body)\n"
                "- Use HTML with <b> for emphasis\n- Match the tone exactly\n"
                "- 80-200 words\n- No placeholders\n- Appropriate sign-off\n\n"
                "Also generate a subject line.\n\n"
                "Format:\nSUBJECT: <subject>\nBODY:\n<body>"
            )
            usr_comp = (
                f"Write a {tone_val} email.\n\nTopic:\n{comp_topic}\n\n"
                f"Recipient: {comp_recipient}\nTone: {tone_val}\n\n"
                "Return SUBJECT: and BODY: as specified."
            )
            resp_comp = llm_comp.invoke([SM3(content=sys_comp), HM3(content=usr_comp)])
            text = resp_comp.content.strip()

            subject_c = ""
            body_c = text
            if "SUBJECT:" in text and "BODY:" in text:
                parts = text.split("BODY:", 1)
                body_c = parts[1].strip() if len(parts) > 1 else text
                subj_p = parts[0].split("SUBJECT:", 1)
                if len(subj_p) > 1:
                    subject_c = subj_p[1].strip().split("\n")[0].strip()
            elif "SUBJECT:" in text:
                tlines = text.split("\n")
                for li, line in enumerate(tlines):
                    if line.strip().startswith("SUBJECT:"):
                        subject_c = line.replace("SUBJECT:", "").strip()
                        body_c = "\n".join(tlines[li + 1:]).strip()
                        break
            if not subject_c:
                subject_c = f"Regarding: {comp_topic[:50]}"

            st.session_state["compose_subject"] = subject_c
            st.session_state["compose_body"] = body_c
            st.session_state["compose_ready"] = True
            st.session_state["compose_generating"] = False
            status.update(label="✅ Email generated!", state="complete")

        st.rerun()

    # Review & Send
    if st.session_state.get("compose_ready"):
        st.markdown("---")
        st.markdown("### 2️⃣ Review & Send")

        ed_subj_c = st.text_input("📌 Subject", value=st.session_state["compose_subject"], key="comp_subj_ed")
        ed_body_c = st.text_area("📝 Email Body", value=st.session_state["compose_body"], height=300, key="comp_body_ed")

        st.markdown("#### 👁️ Email Preview")
        preview_c = ed_body_c.replace("\n", "<br>")
        st.markdown(
            f'<div style="background:#f8f9fa;color:#1a1a2e;padding:1rem 1.2rem;'
            f'border-radius:10px;font-family:Arial;font-size:14px;line-height:1.6;'
            f'border:1px solid #dee2e6;">{preview_c}</div>',
            unsafe_allow_html=True,
        )

        s1, s2, s3 = st.columns(3)
        with s1:
            send_comp = st.button("📤 Send Email", use_container_width=True, key="send_compose")
        with s2:
            regen_comp = st.button("🔄 Regenerate", use_container_width=True, key="regen_compose")
        with s3:
            discard_comp = st.button("🗑️ Discard", use_container_width=True, key="discard_compose")

        if send_comp:
            if not sender_email or not sender_password:
                st.error("❌ Enter email credentials in sidebar.")
            else:
                with st.spinner("📤 Sending email …"):
                    from email_sender import send_composed_email
                    result = send_composed_email(
                        to_email=comp_recipient,
                        subject=ed_subj_c,
                        body=ed_body_c,
                        smtp_server=smtp_server,
                        smtp_port=int(smtp_port),
                        sender_email=sender_email,
                        sender_password=sender_password,
                        cc=comp_cc,
                        bcc=comp_bcc,
                        attachment_paths=st.session_state.get("compose_attachments"),
                    )
                if result["success"]:
                    st.success(f"✅ Email sent to **{comp_recipient}**!")
                    st.balloons()
                else:
                    st.error(f"❌ Failed: {result['error']}")
                st.session_state["compose_ready"] = False
                st.session_state["compose_body"] = ""
                st.session_state["compose_subject"] = ""

        if regen_comp:
            st.session_state["compose_ready"] = False
            st.session_state["compose_body"] = ""
            st.session_state["compose_subject"] = ""
            st.rerun()

        if discard_comp:
            st.session_state["compose_ready"] = False
            st.session_state["compose_body"] = ""
            st.session_state["compose_subject"] = ""
            st.info("🗑️ Discarded.")
