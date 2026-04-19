JOB_DB = {
    "Data Scientist": ["python", "machine learning", "numpy"],
    "Data Analyst": ["excel", "sql", "power bi"],
    "Web Developer": ["html", "css", "javascript"]
}

def recommend_jobs(user_skills):
    recommendations = []

    for job, skills in JOB_DB.items():
        match = len(set(skills) & set(user_skills))

        if match >= 2:
            recommendations.append(job)

    return recommendations if recommendations else ["No strong match found"]