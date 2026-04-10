import google.generativeai as genai

# ✅ Model config
MODEL_NAME = "gemini-1.5-flash"

# ✅ Track config state
_IS_CONFIGURED = False


def configure_gemini(api_key: str) -> None:
    """Configures the Gemini API client with the given key."""
    global _IS_CONFIGURED
    if api_key:
        try:
            genai.configure(api_key=api_key)
            _IS_CONFIGURED = True
        except Exception:
            _IS_CONFIGURED = False


def check_configured() -> bool:
    """Check if API is configured properly."""
    return _IS_CONFIGURED


# ✅ Reuse model (performance improvement)
def _get_model():
    try:
        return genai.GenerativeModel(MODEL_NAME)
    except Exception:
        return None


def _safe_generate(prompt: str) -> str:
    """
    Safe wrapper for Gemini API calls.
    """
    if not check_configured():
        return "⚠️ Gemini API key not configured."

    try:
        model = _get_model()
        if model is None:
            return "⚠️ Model initialization failed."

        response = model.generate_content(prompt)

        if not response or not hasattr(response, "text") or not response.text:
            return "⚠️ Empty response from model."

        return response.text.strip()

    except Exception as e:
        return f"⚠️ API Error: {str(e)}"


def generate_candidate_analysis(resume_text: str, jd: str) -> str:
    """
    Generates a deep analysis of the candidate's resume against the job description.
    """
    prompt = f"""
    You are an expert technical recruiter analyzing a candidate's resume for a specific job description.
    
    Job Description:
    {jd}
    
    Candidate Resume:
    {resume_text}
    
    Provide a comprehensive analysis including:
    1. Recommendation (Hire / Hold / Reject) based on the match.
    2. Deep Gap Analysis: What specific skills or experiences are missing or weakly supported?
    3. Strengths: What are the best matching parts of this resume?
    4. Red Flags: Are there any concerns (e.g., job hopping, missing core requirements)?
    
    Be objective, precise, and concise. Use Markdown formatting.
    """

    return _safe_generate(prompt)


def generate_interview_questions(resume_text: str, jd: str) -> str:
    """
    Generates tailored interview questions for the candidate.
    """
    prompt = f"""
    You are an expert technical interviewer preparing to interview a candidate.
    
    Job Description:
    {jd}
    
    Candidate Resume:
    {resume_text}
    
    Generate 5 specific, tailored interview questions based on strengths and gaps.
    
    Format as a numbered Markdown list.
    """

    return _safe_generate(prompt)


def generate_email_draft(candidate_name: str, resume_text: str, jd: str, status: str) -> str:
    """
    Generates a draft email (acceptance or rejection).
    """
    if status.lower() == "accept":
        intent = "invite them to an interview, mentioning a couple of strengths."
    else:
        intent = "politely reject them with one constructive feedback point."

    prompt = f"""
    You are a technical recruiter drafting an email to a candidate named {candidate_name}.
    
    Job Description:
    {jd}
    
    Candidate Resume:
    {resume_text}
    
    Your goal is to {intent}
    
    Draft a professional, concise email.
    """

    return _safe_generate(prompt)