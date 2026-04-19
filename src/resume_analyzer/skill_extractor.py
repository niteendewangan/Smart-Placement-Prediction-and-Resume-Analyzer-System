import spacy

nlp = spacy.load("en_core_web_sm")

SKILL_DB = {
    "data scientist": ["python", "machine learning", "pandas", "numpy"],
    "web developer": ["html", "css", "javascript"],
    "data analyst": ["excel", "sql", "power bi"]
}

def extract_skills(text):
    text = text.lower()
    found = []

    for role, skills in SKILL_DB.items():
        for skill in skills:
            if skill in text:
                found.append(skill)

    return list(set(found))


def get_resume_score(skills):
    score = len(skills) * 12
    return min(score, 100)


def missing_skills(user_skills):
    all_skills = set(sum(SKILL_DB.values(), []))
    return list(all_skills - set(user_skills))