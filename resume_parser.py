"""
resume_parser.py
────────────────
Extract text from a Resume PDF and produce a concise LLM-generated summary
of skills, experience, and education.
"""

from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


def extract_text_from_pdf(pdf_path: str) -> str:
    """Read every page of a PDF and return the concatenated text."""
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def summarize_resume(raw_text: str, groq_api_key: str) -> str:
    """
    Send the raw resume text to ChatGroq and get back a structured summary
    suitable for email personalisation.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.3,
        max_tokens=600,
    )

    system_prompt = (
        "You are a resume-analysis assistant. Given the raw text of a resume, "
        "produce a detailed summary (≤ 300 words) covering:\n"
        "1. Full name\n"
        "2. Key technical skills — separate Full Stack (languages, frameworks, databases) "
        "and AI/ML skills (models, frameworks like TensorFlow, PyTorch, LangChain, etc.)\n"
        "3. ALL projects listed on the resume — include name and 1-line description each\n"
        "4. Education\n"
        "5. Any notable achievements or certifications\n\n"
        "Make sure to capture EVERY project and skill — do not skip any.\n"
        "Output ONLY the summary — no preamble."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Resume text:\n\n{raw_text}"),
    ]

    response = llm.invoke(messages)
    return response.content.strip()


def parse_resume(pdf_path: str, groq_api_key: str) -> dict:
    """
    High-level helper: extract text → summarise → return both.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    summary = summarize_resume(raw_text, groq_api_key)
    return {"raw_text": raw_text, "summary": summary}
