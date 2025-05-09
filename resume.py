import os
import fitz  # PyMuPDF
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
import re
from werkzeug.utils import secure_filename
import docx  # For DOCX files
import chardet  # For detecting file encoding

# Initialize Flask app
app = Flask(__name__)

# Ensure uploads directory exists
os.makedirs('uploads', exist_ok=True)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Define job role keywords dictionary - we'll use this for role matching
JOB_ROLE_KEYWORDS = {
    "Software Developer": ["python", "java", "javascript", "software development", "programming", "coding", 
                          "algorithm", "git", "api", "database", "web development", "backend", "frontend"],
    "Data Scientist": ["python", "r", "machine learning", "data analysis", "statistics", "sql", "data visualization", 
                      "pandas", "numpy", "tensorflow", "scikit-learn", "big data", "data mining"],
    "Project Manager": ["project management", "agile", "scrum", "team lead", "leadership", "planning", 
                       "coordination", "stakeholder", "pmp", "budget", "timeline", "resource management"],
    "UI/UX Designer": ["ui", "ux", "user interface", "user experience", "figma", "sketch", "adobe xd", 
                      "wireframing", "prototyping", "usability", "design thinking", "typography"],
    "Marketing Specialist": ["marketing", "social media", "seo", "content strategy", "analytics", "campaign", 
                           "digital marketing", "brand", "advertising", "market research", "conversion"],
    "Financial Analyst": ["finance", "accounting", "financial analysis", "excel", "forecasting", "budgeting", 
                         "financial modeling", "reporting", "valuation", "investment", "risk assessment"],
    "Network Engineer": ["networking", "cisco", "routing", "switching", "tcp/ip", "network security", 
                        "firewall", "lan", "wan", "vpn", "cloud infrastructure", "network troubleshooting"]
}

# Function to Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Function to Extract Text from DOCX
def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

# Function to Extract Text from TXT
def extract_text_from_txt(txt_path):
    try:
        # Detect the encoding of the text file
        with open(txt_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        # Read the file with the detected encoding
        with open(txt_path, 'r', encoding=encoding) as file:
            return file.read()
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        return ""

# Function to Extract Text from RTF
def extract_text_from_rtf(rtf_path):
    try:
        # This is a simple approach using a third-party library or command-line tool
        # For full implementation, you may want to use a library like 'striprtf' or 'pyth'
        # Here's a simple implementation
        import subprocess
        
        try:
            # Try using 'unrtf' command line tool if available
            result = subprocess.run(['unrtf', '--text', rtf_path], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
        except:
            pass
        
        # Fallback to basic approach - strip RTF codes (simplified)
        with open(rtf_path, 'r', encoding='utf-8', errors='ignore') as file:
            rtf_text = file.read()
            
        # Very basic RTF stripping (for complete solution use a dedicated RTF library)
        text = re.sub(r'\\[a-z]+', ' ', rtf_text)  # Remove RTF commands
        text = re.sub(r'\{.*?\}', '', text)  # Remove RTF groups
        text = re.sub(r'\\[^a-z]', '', text)  # Remove escaped chars
        return text
    except Exception as e:
        print(f"Error extracting text from RTF: {e}")
        return ""

# General function to extract text based on file extension
def extract_text_from_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    elif file_extension == '.rtf':
        return extract_text_from_rtf(file_path)
    else:
        return ""

# Function to Preprocess Text
def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])


# Enhanced function to extract skills more specifically
def extract_skills(text):
    # Common programming languages and technologies
    programming_skills = [
        "python", "java", "javascript", "c++", "c#", "ruby", "php", "swift", "kotlin", "golang",
        "react", "angular", "vue", "node.js", "django", "flask", "spring", "tensorflow", "pytorch",
        "sql", "mysql", "postgresql", "mongodb", "oracle", "nosql", "aws", "azure", "gcp",
        "docker", "kubernetes", "jenkins", "git", "html", "css", "rest api", "graphql"
    ]
    
    # Common soft skills
    soft_skills = [
        "leadership", "communication", "teamwork", "problem solving", "critical thinking",
        "time management", "creativity", "adaptability", "negotiation", "decision making",
        "project management", "agile", "scrum", "presentation", "analytical"
    ]
    
    # Combine all skills
    all_skills = programming_skills + soft_skills
    
    # Find matches in the text (case insensitive)
    found_skills = []
    text_lower = text.lower()
    
    for skill in all_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.append(skill.capitalize())  # Capitalize for better display
    
    return found_skills


# Function to Extract Named Entities (e.g., skills, job titles, companies, education)
def extract_named_entities(text):
    doc = nlp(text)
    skills = extract_skills(text)  # Use our enhanced skill extraction
    education = []
    job_titles = []
    organizations = []
    
    # Define the entities we care about
    for ent in doc.ents:
        if ent.label_ == "ORG":
            organizations.append(ent.text)
        elif ent.label_ == "GPE" or ent.label_ == "LOC":
            education.append(ent.text)  # This can be adapted to capture institutions or degrees
        elif ent.label_ == "WORK_OF_ART" or ent.label_ == "PERSON":
            job_titles.append(ent.text)

    # Look for education keywords
    education_keywords = ["bachelor", "master", "phd", "doctorate", "degree", "bs", "ba", "msc"]
    for keyword in education_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
            # Extract the sentence containing the education keyword
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                if keyword in sentence.lower():
                    education.append(sentence.strip())

    return {
        "skills": skills,
        "education": list(set(education)),  # Remove duplicates
        "job_titles": list(set(job_titles)),  # Remove duplicates
        "organizations": list(set(organizations))  # Remove duplicates
    }


# New function to suggest job roles based on resume content
def suggest_job_roles(resume_info):
    skills = [skill.lower() for skill in resume_info["skills"]]
    job_titles = [title.lower() for title in resume_info["job_titles"]]
    
    # Calculate match scores for each job role
    role_scores = {}
    for role, keywords in JOB_ROLE_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            # Check if the keyword is in skills
            if keyword in skills:
                score += 2  # Higher weight for skills match
            
            # Check if the keyword is in job titles
            for title in job_titles:
                if keyword in title:
                    score += 1
        
        role_scores[role] = score
    
    # Sort roles by score and return top 3
    sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Only return roles with non-zero scores, up to 3
    suggested_roles = [role for role, score in sorted_roles if score > 0][:3]
    
    # If no roles match, suggest "General"
    if not suggested_roles:
        suggested_roles = ["General Position"]
    
    return suggested_roles


# Function to extract key requirements from job description
def extract_job_requirements(job_description):
    doc = nlp(job_description)
    skills_required = extract_skills(job_description)
    
    # Look for years of experience
    experience_pattern = re.compile(r'(\d+)[\+\s-]*years?[\s\w]*experience', re.IGNORECASE)
    experience_matches = experience_pattern.findall(job_description)
    experience_required = [int(years) for years in experience_matches] if experience_matches else []
    
    # Look for education requirements
    education_required = []
    education_keywords = ["bachelor", "master", "phd", "doctorate", "degree", "bs", "ba", "msc"]
    for keyword in education_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', job_description.lower()):
            sentences = re.split(r'[.!?]', job_description)
            for sentence in sentences:
                if keyword in sentence.lower():
                    education_required.append(sentence.strip())
                    break
    
    return {
        "skills_required": skills_required,
        "experience_required": experience_required,
        "education_required": education_required
    }


# Function to Vectorize Text using TF-IDF
def vectorize_text(job_description, resumes):
    vectorizer = TfidfVectorizer()
    all_text = [job_description] + resumes
    tfidf_matrix = vectorizer.fit_transform(all_text)
    return tfidf_matrix


# Function to Calculate Cosine Similarity
def calculate_cosine_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])


# Function to Rank Resumes Based on Similarity
def rank_resumes(cosine_similarities):
    similarity_scores = cosine_similarities[0]
    ranked_resumes = np.argsort(similarity_scores)[::-1]
    return ranked_resumes, similarity_scores


# Route for the Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Route for Handling Resume Uploads - changing to match frontend API endpoint
@app.route('/api/analyze-resumes', methods=['POST'])
def analyze_resumes():
    if 'resume1' not in request.files or 'resume2' not in request.files:
        return jsonify({'error': 'Missing resume files'}), 400
    
    resume1 = request.files['resume1']
    resume2 = request.files['resume2']
    
    # Get the job description from the form
    job_description = request.form.get('job_description', "Looking for a software developer with knowledge of Python, Java, and React.")
    
    # Save the files temporarily with secure filenames
    resume1_path = os.path.join('uploads', secure_filename(resume1.filename))
    resume2_path = os.path.join('uploads', secure_filename(resume2.filename))
    
    resume1.save(resume1_path)
    resume2.save(resume2_path)
    
    try:
        # Extract text from resumes using our new multi-format function
        resume1_text = extract_text_from_file(resume1_path)
        resume2_text = extract_text_from_file(resume2_path)
        
        if not resume1_text or not resume2_text:
            return jsonify({'error': 'Failed to extract text from one or both resumes'}), 400
        
        # Preprocess the resumes
        processed_resume1 = preprocess_text(resume1_text)
        processed_resume2 = preprocess_text(resume2_text)
        
        # Extract useful information from resumes
        resume1_info = extract_named_entities(resume1_text)
        resume2_info = extract_named_entities(resume2_text)
        
        # Suggest job roles based on resume content
        resume1_roles = suggest_job_roles(resume1_info)
        resume2_roles = suggest_job_roles(resume2_info)
        
        # Extract requirements from job description
        job_requirements = extract_job_requirements(job_description)
        
        # Vectorize the job description and resumes
        tfidf_matrix = vectorize_text(job_description, [processed_resume1, processed_resume2])
        
        # Calculate cosine similarity
        cosine_similarities = calculate_cosine_similarity(tfidf_matrix)
        
        # Rank the resumes
        ranked_resumes, similarity_scores = rank_resumes(cosine_similarities)
        
        # Determine which resume is more suitable for the job
        more_suitable_resume = "Resume 1" if ranked_resumes[0] == 0 else "Resume 2"
        
        # Detailed skill match analysis
        resume1_skills = set(resume1_info["skills"])
        resume2_skills = set(resume2_info["skills"])
        required_skills = set(job_requirements["skills_required"])
        
        resume1_skill_match = resume1_skills.intersection(required_skills)
        resume2_skill_match = resume2_skills.intersection(required_skills)
        
        resume1_match_percentage = len(resume1_skill_match) / len(required_skills) * 100 if required_skills else 0
        resume2_match_percentage = len(resume2_skill_match) / len(required_skills) * 100 if required_skills else 0
        
        # Clean up temporary files
        try:
            os.remove(resume1_path)
            os.remove(resume2_path)
        except:
            pass
        
        # Return JSON response for the frontend
        return jsonify({
            'job': {
                'description': job_description,
                'requirements': {
                    'skills': job_requirements["skills_required"],
                    'experience': job_requirements["experience_required"][0] if job_requirements["experience_required"] else 0,
                    'education': job_requirements["education_required"]
                }
            },
            'resume1': {
                'match_percentage': round(resume1_match_percentage),
                'skill_match': list(resume1_skill_match),
                'skills': resume1_info["skills"],
                'education': resume1_info["education"],
                'job_titles': resume1_info["job_titles"],
                'organizations': resume1_info["organizations"],
                'suggested_roles': resume1_roles
            },
            'resume2': {
                'match_percentage': round(resume2_match_percentage),
                'skill_match': list(resume2_skill_match),
                'skills': resume2_info["skills"],
                'education': resume2_info["education"],
                'job_titles': resume2_info["job_titles"],
                'organizations': resume2_info["organizations"],
                'suggested_roles': resume2_roles
            }
        })
        
    except Exception as e:
        # Clean up temporary files in case of error
        try:
            os.remove(resume1_path)
            os.remove(resume2_path)
        except:
            pass
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500


# Route for Result Page
@app.route('/result')
def result():
    # This is needed for direct access to result page, but we'll mostly be using the JSON API
    return render_template('result.html')


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)