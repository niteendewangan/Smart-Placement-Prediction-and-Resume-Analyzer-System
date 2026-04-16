import spacy

nlp = spacy.load("en_core_web_sm")

SKILLS = [
    "python", "java", "c++", "sql", "machine learning",
    "data analysis", "deep learning", "excel", "power bi"
]

def extract_skills(text):
    doc = nlp(text)
    found_skills = []

    for skill in SKILLS:
        if skill in text:
            found_skills.append(skill)

    return list(set(found_skills))


def get_resume_score(skills):
    score = len(skills) * 10
    return min(score, 100)