"""
research_tool.py
────────────────
Search the internet for basic company information using Tavily
and return a structured summary via LLM.
"""

from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


def search_company(
    company_name: str,
    industry: str,
    location: str,
    tavily_api_key: str,
) -> list[dict]:
    """
    Use Tavily to search for company information.
    Returns a list of result dicts with 'title', 'url', 'content'.
    """
    client = TavilyClient(api_key=tavily_api_key)
    query = f"{company_name} company {industry} {location} what do they do"
    response = client.search(query=query, max_results=5)
    return response.get("results", [])


def summarize_company(
    company_name: str,
    search_results: list[dict],
    groq_api_key: str,
) -> str:
    """
    Feed raw search results into ChatGroq to get a short, structured
    company summary for email personalisation.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.3,
        max_tokens=400,
    )

    combined_content = "\n\n".join(
        f"Source: {r.get('title', 'N/A')}\n{r.get('content', '')}"
        for r in search_results
    )

    system_prompt = (
        "You are a research assistant. Given search results about a company, "
        "produce a concise summary (≤ 100 words) covering:\n"
        "1. What the company does\n"
        "2. Industry / domain\n"
        "3. Any recent news or notable projects\n"
        "4. Company culture or values (if found)\n\n"
        "Output ONLY the summary."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Company: {company_name}\n\n"
                f"Search results:\n{combined_content}"
            )
        ),
    ]

    response = llm.invoke(messages)
    return response.content.strip()


def research_company(
    company_name: str,
    industry: str,
    location: str,
    tavily_api_key: str,
    groq_api_key: str,
) -> str:
    """
    End-to-end: Tavily search → LLM summarise → return company context string.
    """
    results = search_company(company_name, industry, location, tavily_api_key)
    if not results:
        return (
            f"{company_name} is a company in the {industry} industry "
            f"based in {location}. No additional details were found online."
        )
    return summarize_company(company_name, results, groq_api_key)
