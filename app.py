import sqlite3
import re
import time
import sys
from datetime import datetime
from functools import wraps
from urllib.parse import urljoin
import markdown   # at the top of your file
import requests
from bs4 import BeautifulSoup
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify
)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from werkzeug.security import generate_password_hash, check_password_hash
import json
import io, base64
import os
from dotenv import load_dotenv

# Text mining metrics libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# Comment out spaCy import since it's not installed
# import spacy
import textstat
from textblob import TextBlob

# Load environment variables from the .env file
load_dotenv()

# Access your Gemini API key from the environment variable
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    print("WARNING: GEMINI_API_KEY not found in environment variables. Summarization will use local fallback methods.")

# Gemini API endpoint configuration
gemini_api_base = "https://generativelanguage.googleapis.com/v1beta"
gemini_model = "gemini-2.0-flash-lite"

txt = ""
# --- CONFIGURATION ---
app = Flask(__name__)
app.secret_key = "YOUR_SECRET_KEY_HERE"  # ← change this!

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9"
}

# --- DATABASE SETUP ---
conn = sqlite3.connect("app.db", check_same_thread=False)
c = conn.cursor()

# Create tables if they don't exist
c.execute("""
CREATE TABLE IF NOT EXISTS scrapes (
    id INTEGER PRIMARY KEY,
    domain TEXT,
    agreement_type TEXT,  -- 'tos' or 'pp'
    url TEXT,            -- URL of the document
    text_length INTEGER,  -- Length of the document (word count)
    summary_100 TEXT,    -- Detailed summary
    summary_25 TEXT,     -- Short summary
    text_metrics TEXT,   -- All text mining metrics stored as JSON
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(domain, agreement_type)
)
""")

# Create summary cache table
c.execute("""
CREATE TABLE IF NOT EXISTS summary_cache (
    id INTEGER PRIMARY KEY,
    tos_hash TEXT,
    pp_hash TEXT,
    tos_summary_100 TEXT,
    tos_summary_25 TEXT,
    pp_summary_100 TEXT,
    pp_summary_25 TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tos_hash, pp_hash)
)
""")

# Create users table for authentication
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    password TEXT NOT NULL
)
""")

conn.commit()

# --- HELPERS ---
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def get_full_url(base_url, link):
    return urljoin(base_url, link)

def find_policy_links(base_url):
    try:
        response = requests.get("https://" + base_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", href=True)

        # Enhanced patterns for Terms of Service matching
        tos_patterns = [
            "terms of service", "terms of use", "terms and conditions",
            "terms & conditions", "user agreement", "service agreement",
            "user terms", "legal terms", "legal agreement", "platform terms",
            "website terms", "terms and policies", "platform agreement",
            "site terms", "service terms", "software license", "user terms",
            "license agreement", "terms agreement", "conditions of use", 
            "condition of use", "terms of sale", "site agreement",
            "legal notices", "legal information", "website agreement",
            "user license", "tos", "tou", "eula"
        ]
        
        # Enhanced patterns for Privacy Policy matching
        pp_patterns = [
            "privacy policy", "privacy statement", "privacy notice",
            "data privacy", "data protection", "privacy protection",
            "privacy practices", "privacy rights", "privacy",
            "your privacy", "personal data", "personal information",
            "data rights", "privacy settings", "data protection rights",
            "right to access", "right to rectification", "right to erasure",
            "right to be forgotten", "do not sell my data", "opt-out rights",
            "ccpa rights", "gdpr rights", "data subject rights"
        ]
        
        # Candidate links with scores
        tos_candidates = []
        pp_candidates = []
        
        for link in links:
            href = link["href"].lower()
            text = link.get_text().lower().strip()
            
            # Skip empty, javascript, and hash links
            if not href or href.startswith(("javascript:", "#", "mailto:", "tel:")):
                continue
                
            # Score for TOS link
            tos_score = 0
            for pattern in tos_patterns:
                if pattern in href:
                    tos_score += 10
                if pattern in text:
                    tos_score += 15
                    
            # Special case boosts for TOS
            if href == "/terms" or href == "/tos" or href == "/terms-of-service":
                tos_score += 30
            if text == "terms of service" or text == "terms of use":
                tos_score += 30
            
            # Score for PP link
            pp_score = 0
            for pattern in pp_patterns:
                if pattern in href:
                    pp_score += 10
                if pattern in text:
                    pp_score += 15
                    
            # Special case boosts for PP
            if href == "/privacy" or href == "/privacy-policy":
                pp_score += 30
            if text == "privacy policy" or text == "privacy":
                pp_score += 30
                
            # Footer links get a bonus (often contain legal links)
            is_footer_link = False
            parent = link.parent
            for _ in range(3):  # Check up to 3 levels of parents
                if not parent:
                    break
                if parent.name == 'footer' or (parent.get('class') and 
                                              any('footer' in cls.lower() for cls in parent.get('class'))):
                    is_footer_link = True
                    break
                parent = parent.parent
                
            if is_footer_link:
                tos_score += 15
                pp_score += 15
                
            # Add to candidates if score is above threshold
            if tos_score >= 10:
                tos_candidates.append({
                    "url": href,
                    "score": tos_score,
                    "text": text
                })
                
            if pp_score >= 10:
                pp_candidates.append({
                    "url": href,
                    "score": pp_score,
                    "text": text
                })
        
        # Sort candidates by score
        tos_candidates.sort(key=lambda x: x["score"], reverse=True)
        pp_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Get top candidates
        tos_link = ""
        if tos_candidates:
            tos_link = get_full_url("https://" + base_url, tos_candidates[0]["url"])
            
        pp_link = ""
        if pp_candidates:
            pp_link = get_full_url("https://" + base_url, pp_candidates[0]["url"])

        return tos_link, pp_link
    except Exception as e:
        print(f"[!] Error fetching {base_url}: {e}")
        return "", ""

def fetch_text(url):
    """ Fetch all <p> text from url """
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        
        # Check for Cloudflare or CAPTCHA related content
        cloudflare_captcha_terms = [
            'cloudflare', 'cloud flare', 'ddos protection', 
            'captcha', 'security check', 'browser check',
            'browser verification', 'challenge page', 'ray id',
            'attention required', 'site checking', 'verify you are human',
            'robot verification', 'please wait while we verify'
        ]
        
        for term in cloudflare_captcha_terms:
            if term.lower() in text.lower():
                raise ValueError(f"Cloudflare or CAPTCHA protection detected: '{term}' found on the page")
                
        return text
    except Exception as e:
        print(f"[!] Error scraping {url}: {e}")
        return ""

def clean_tokens(text):
    stops = set(stopwords.words('english'))
    t = re.sub(r'[^a-z\s]', '', text.lower())
    return [w for w in t.split() if w and w not in stops]

def call_gemini_api(prompt):
    """Make a call to the Gemini API"""
    # Load environment variables from the .env file to ensure we have the latest API key
    load_dotenv()
    
    # Get API key from environment
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
        
    # Set up the API endpoint and headers
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={gemini_api_key}"
    api_headers = {
        "Content-Type": "application/json"
    }
    
    # Request payload
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.4,
            "topP": 0.95,
            "maxOutputTokens": 1000
        }
    }
    
    # Make the API request
    response = requests.post(url, headers=api_headers, json=data)
    
    # Check response status
    if response.status_code == 200:
        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            content = result["candidates"][0]["content"]
            return content["parts"][0]["text"]
        else:
            print("Unexpected response format from Gemini API")
            print(f"Response: {json.dumps(result, indent=2)}")
            return None
    else:
        print(f"Error calling Gemini API: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def summarize_tos_pp(tos_text, pp_text, company_name=None):
    # First check if we have a cached result for this exact content
    tos_hash = hash(tos_text[:1000] if tos_text else "")  # Use first 1000 chars as hash key
    pp_hash = hash(pp_text[:1000] if pp_text else "")
    
    # Check if we have cached results in the database
    c.execute("""
        SELECT tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25 
        FROM summary_cache 
        WHERE tos_hash = ? AND pp_hash = ?
    """, (tos_hash, pp_hash))
    
    cached = c.fetchone()
    if cached:
        return cached[0], cached[1], cached[2], cached[3]
    
    # If no content to summarize, return empty strings
    if not tos_text and not pp_text:
        return "No terms of service found", "No TOS found", "No privacy policy found", "No PP found"
    
    max_retries = 2  # Try up to 2 more times after the initial attempt
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Check if API key exists
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY not set in environment variables")
                
            # Use the improved prompt from the create_improved_summary_prompt function
            if tos_text:
                prompt = create_improved_summary_prompt(tos_text, "tos", company_name)
            elif pp_text:
                prompt = create_improved_summary_prompt(pp_text, "pp", company_name)
            else:
                raise ValueError("No text provided for summarization")
                
            # Call Gemini API
            full_txt = call_gemini_api(prompt)
            if not full_txt:
                raise ValueError("Empty response from Gemini API")
          
            # Split into clean lines
            lines = [line.strip() for line in full_txt.splitlines() if line.strip()]
            
            # Extract summaries using the new format from the improved prompt
            tos_summary_100 = ""
            tos_summary_25 = ""
            pp_summary_100 = ""
            pp_summary_25 = ""
            
            # Extract the 100-word summary and one-sentence summary
            hundred_word_start = full_txt.find("100-WORD SUMMARY")
            one_sentence_start = full_txt.find("ONE-SENTENCE SUMMARY")
            if hundred_word_start >= 0 and one_sentence_start >= 0:
                # Extract the section between the headers, skipping the header line
                hundred_word_text = full_txt[hundred_word_start:one_sentence_start].strip()
                hundred_word_lines = hundred_word_text.split('\n')
                hundred_word_summary = '\n'.join(hundred_word_lines[1:]).strip()
                
                # Extract one-sentence summary
                one_sentence_text = full_txt[one_sentence_start:].strip()
                one_sentence_lines = one_sentence_text.split('\n')
                one_sentence_summary = '\n'.join(one_sentence_lines[1:]).strip()
                
                # Find any requirements section and remove it (usually appears after a blank line)
                if "Requirements:" in hundred_word_summary:
                    hundred_word_summary = hundred_word_summary.split("Requirements:")[0].strip()
                if "Requirements:" in one_sentence_summary:
                    one_sentence_summary = one_sentence_summary.split("Requirements:")[0].strip()
                    
                # Clean up the text (remove markdown formatting, extra spaces)
                hundred_word_summary = re.sub(r'[*"`\']+', '', hundred_word_summary)
                one_sentence_summary = re.sub(r'[*"`\']+', '', one_sentence_summary)
                
                # Assign to the appropriate variables based on the document type
                if tos_text:
                    tos_summary_100 = hundred_word_summary
                    tos_summary_25 = one_sentence_summary
                elif pp_text:
                    pp_summary_100 = hundred_word_summary
                    pp_summary_25 = one_sentence_summary
            
            # Check if any required summary is empty
            is_tos_empty = tos_text and (not tos_summary_100 or not tos_summary_25)
            is_pp_empty = pp_text and (not pp_summary_100 or not pp_summary_25)
            
            if is_tos_empty or is_pp_empty:
                # If we still have retries left, try again
                if retry_count < max_retries:
                    retry_count += 1
                    print(f"Empty summary detected, retrying ({retry_count}/{max_retries})...")
                    time.sleep(1)  # Add a small delay before retrying
                    continue
                else:
                    # We've exhausted all retries and still have empty summaries
                    empty_summary_type = "TOS" if is_tos_empty else "PP"
                    raise ValueError(f"Failed to generate {empty_summary_type} summary after multiple attempts")
            
            # If we got here, summaries are not empty, so we can cache and return the results
            c.execute("""
                INSERT INTO summary_cache 
                (tos_hash, pp_hash, tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (tos_hash, pp_hash, tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25))
            
            conn.commit()
            
            return tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25
            
        except Exception as e:
            if retry_count < max_retries:
                retry_count += 1
                print(f"Error in summarization, retrying ({retry_count}/{max_retries}): {e}")
                time.sleep(1)  # Add a small delay before retrying
                continue
            else:
                print(f"Error in summarization after {retry_count} retries: {e}")
                # No fallback methods - just return empty values with error
                return "", f"Error: {str(e)}", f"Error: {str(e)[:50]}", f"Error: {str(e)[:50]}"

def analyze_tos_pp(tos_text, pp_text, company_name=None):
    # clean & tokens
    tok_tos = clean_tokens(tos_text)
    tok_pp  = clean_tokens(pp_text)
    freq_tos = Counter(tok_tos).most_common(10)
    freq_pp  = Counter(tok_pp).most_common(10)

    tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25 = summarize_tos_pp(tos_text, pp_text, company_name)
    return freq_tos, freq_pp, tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25

def calculate_text_metrics(tos_text, pp_text):
    metrics = {}
    
    def analyze_single_document(text, doc_type, other_text=None):
        if not text:
            return {
                'tfidf': [],
                'readability': {'flesch_ease': 0, 'flesch_grade': 0, 'gunning_fog': 0},
                'sentiment': {'polarity': 0, 'subjectivity': 0},
                'doc_stats': {'words': 0, 'sentences': 0, 'avg_length': 0},
                'hapax': {'ratio': 0, 'count': 0, 'examples': []}
            }
            
        # 1. TF-IDF Analysis with better preprocessing
        try:
            # Create a mini-corpus for TF-IDF
            mini_corpus = [text]
            if other_text:
                mini_corpus.append(other_text)

            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(mini_corpus)
            feature_names = vectorizer.get_feature_names_out()

            # Get top TF-IDF scores for the target document
            doc_tfidf = tfidf_matrix[0].toarray()[0]
            top_indices = np.argsort(doc_tfidf)[::-1][:10]
            top_tfidf = [(feature_names[i], float(doc_tfidf[i])) for i in top_indices]
        except ValueError:
            top_tfidf = []
            
        # 2. Readability Analysis
        try:
            readability = {
                'flesch_ease': textstat.flesch_reading_ease(text),
                'flesch_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text)
            }
        except:
            readability = {'flesch_ease': 0, 'flesch_grade': 0, 'gunning_fog': 0}
            
        # 3. Sentiment Analysis with TextBlob
        try:
            blob = TextBlob(text)
            sentiment = {
                'polarity': round(blob.sentiment.polarity, 3),
                'subjectivity': round(blob.sentiment.subjectivity, 3)
            }
        except:
            sentiment = {'polarity': 0, 'subjectivity': 0}
            
        # 4. Document Length Metrics
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        # filter out non-alphanumeric tokens
        clean_words = [w for w in words if w.isalnum()]
        num_words = len(clean_words)
        num_sentences = len(sentences)
        avg_len = num_words / num_sentences if num_sentences > 0 else 0
        doc_stats = {
            'words': num_words,
            'sentences': num_sentences,
            'avg_length': round(avg_len, 2)
        }

        # 5. Hapax Legomena Ratio
        wc = Counter(clean_words)
        hapax_words = [w for w,c in wc.items() if c == 1]
        hapax_ratio = len(hapax_words) / num_words if num_words else 0
        hapax = {
            'ratio': round(hapax_ratio, 4),
            'count': len(hapax_words),
            'examples': hapax_words[:10]
        }

        return {
            'tfidf': top_tfidf,
            'readability': readability,
            'sentiment': sentiment,
            'doc_stats': doc_stats,
            'hapax': hapax
        }
    
    # Analyze both documents
    metrics['tos'] = analyze_single_document(tos_text, 'tos', pp_text)
    metrics['pp'] = analyze_single_document(pp_text, 'pp', tos_text)
    
    return metrics

# --- Simplified Text Mining Metrics ---
def calculate_simple_metrics(text):
    """
    Compute five lightweight metrics for a single document:
    TF-IDF top terms, Flesch score, sentiment polarity,
    document length stats, and Hapax Legomena ratio.
    """
    if not text:
        return {'tfidf': [], 'readability': 0, 'sentiment': 0,
                'word_count': 0, 'sentence_count': 0,
                'avg_sentence_length': 0, 'hapax_ratio': 0}
    # TF-IDF Top 5 terms
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    tfidf_matrix = vectorizer.fit_transform([text])
    features = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    top_idx = np.argsort(scores)[::-1][:5]
    top_tfidf = [(features[i], float(scores[i])) for i in top_idx]
    # Readability
    flesch = textstat.flesch_reading_ease(text)
    # Sentiment
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    # Document length metrics
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    clean_words = [w for w in words if w.isalnum()]
    wcount = len(clean_words)
    scount = len(sentences)
    avg_len = round(wcount/scount, 2) if scount else 0
    # Hapax Legomena ratio
    wc = Counter(clean_words)
    hapax_count = len([w for w,c in wc.items() if c==1])
    hapax_ratio = round(hapax_count/wcount, 4) if wcount else 0
    return {
        'tfidf': top_tfidf,
        'readability': round(flesch, 1),
        'sentiment': polarity,
        'word_count': wcount,
        'sentence_count': scount,
        'avg_sentence_length': avg_len,
        'hapax_ratio': hapax_ratio,
        # Top 10 word frequencies (filtered tokens)
        'word_frequency': Counter(clean_tokens(text)).most_common(10)
    }

def extract_company_name(domain):
    """
    Extract a company name from a domain URL
    """
    # Remove common TLDs and subdomains
    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    
    # Extract the main domain name (before the TLD)
    parts = domain.split('.')
    if len(parts) > 0:
        company = parts[0]
        # Capitalize the first letter of each word
        company = ' '.join(word.capitalize() for word in company.split('-'))
        # Remove common URL indicators
        return company
    return domain

def create_improved_summary_prompt(text, document_type="tos", company_name=None):
    """
    Creates an improved prompt for generating text summaries using AI models.
    Based on production-grade prompt engineering techniques.
    
    Args:
        text: The document text to summarize
        document_type: 'tos' or 'pp' or other document type
        company_name: Optional name of company for more specific prompt
        
    Returns:
        A formatted prompt string for AI summarization
    """
    # Map document type to full name
    if document_type == "pp":
        doc_type_full = "Privacy Policy"
    elif document_type == "tos":
        doc_type_full = "Terms of Service"
    else:
        # Default to using the original value
        doc_type_full = document_type
    
    # Include company name if available
    company_reference = ""
    company_start = ""
    if company_name:
        company_reference = f" for {company_name}"
        company_start = f"{company_name}'s "
    
    # Construct the prompt
    prompt = f"""100-WORD SUMMARY

Write a concise, factual 100-word summary of the {doc_type_full}{company_reference}. Focus on the company policies and practices without referencing external services, other companies, or general industry practices.

Requirements:
- Start with "{company_start}{doc_type_full}"
- Exactly 100 words (±10)
- Single paragraph
- Objective, factual tone
- No personal pronouns (I, we, you)
- No meta-references (e.g., "this document", "this text" ,"this policy")
- No conditional language (e.g., "may", "might", "could")
- No links or external references

Provide a direct, factual, and company-specific summary.


ONE-SENTENCE SUMMARY

Write a single sentence (maximum 40 words) summarizing the most important aspect of the {doc_type_full}{company_reference}. Focus on the company policies and practices without referencing external services, other companies, or general industry practices.

Requirements:
- Start with "{company_start}{doc_type_full}"
- One clear, direct sentence
- Maximum 40 words
- Objective, factual tone
- No personal pronouns (I, we, you)
- No meta-references (e.g., "this document", "this text")
- No conditional language (e.g., "may", "might", "could")
- No links or external references


Here is the document content:

{text}"""
    
    return prompt

# --- ROUTES ---
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method=="POST":
        u = request.form["username"]
        p = generate_password_hash(request.form["password"])
        try:
            c.execute("INSERT INTO users(username,password) VALUES(?,?)",(u,p))
            conn.commit()
            flash("Registered successfully! Please log in.","success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username taken.","danger")
    return render_template("register.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        u = request.form["username"]
        p = request.form["password"]
        c.execute("SELECT id,password FROM users WHERE username=?",(u,))
        row = c.fetchone()
        if row and check_password_hash(row[1],p):
            session["user_id"]=row[0]
            return redirect(url_for("index"))
        flash("Invalid credentials.","danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/", methods=["GET","POST"])
def index():
    results = []
    # Persist the selected agreement type for form
    selected_type = 'tos'
    if request.method=="POST":
        # Get domains and selected agreement type
        doms = request.form["domains"]
        selected_type = request.form.get("agreement_type", "tos")
        agreement_type = selected_type
        
        for d in [x.strip() for x in doms.split(",") if x.strip()]:
            start_time = time.time()  # Start timing
            
            # Check if the domain and agreement_type combination already exists in the database
            c.execute("SELECT id FROM scrapes WHERE domain = ? AND agreement_type = ?", (d, agreement_type))
            existing_record = c.fetchone()
            
            if existing_record:
                # If the record already exists, retrieve it instead of creating a new one
                c.execute("SELECT * FROM scrapes WHERE id = ?", (existing_record[0],))
                record = c.fetchone()
                
                # Extract company name and prefix summaries
                company_name = extract_company_name(d)
                company_prefix = f"{company_name}'s " if company_name else ""
                
                # Format the result to display
                text_metrics = eval(record[7])  # Convert the string back to a dictionary
                
                # Get the summaries and add company name if not already present
                summary_100 = record[5]
                summary_25 = record[6]
                
                # Add company prefix if not already present
                if summary_100 and not summary_100.startswith(company_prefix):
                    # Remove "The " from the beginning if present
                    if summary_100.startswith("The "):
                        summary_100 = summary_100[4:]
                    summary_100 = company_prefix + summary_100
                    
                if summary_25 and not summary_25.startswith(company_prefix):
                    if summary_25.startswith("The "):
                        summary_25 = summary_25[4:]
                    summary_25 = company_prefix + summary_25
                
                # Calculate processing time (minimal since we're just retrieving)
                processing_time = time.time() - start_time
                time_str = f"{processing_time*1000:.0f}ms"
                
                # Add retrieved result to display list
                result = {
                    "domain": record[1],
                    "agreement_type": record[2],
                    "url": record[3],
                    "text_length": record[4],
                    "summary_100": summary_100,
                    "summary_25": summary_25,
                    "text_metrics": text_metrics,
                    "processing_time": time_str,
                    "note": "Retrieved from database (already exists)"
                }
                
                results.append(result)
                continue
            
            # All-or-nothing approach: only save to DB if everything succeeds
            try:
                # Step 1: Find policy links
                tos_url, pp_url = find_policy_links(d)
                
                # Select URL based on agreement type
                url = tos_url if agreement_type == 'tos' else pp_url
                if not url:
                    doc_type = 'Terms of Service' if agreement_type=='tos' else 'Privacy Policy'
                    raise ValueError(f"No valid {doc_type} URL found for '{d}'")
                
                # Step 2: Fetch and analyze text
                text = fetch_text(url)
                if text.startswith("https"):
                    text = fetch_text(text)
                
                if not text:
                    raise ValueError(f"No content found at {url}")
                
                # Step 3: Extract company name from domain
                company_name = extract_company_name(d)
                    
                # Step 4: Calculate summaries based on agreement type
                if agreement_type == "tos":
                    summary_100, summary_25, _, _ = summarize_tos_pp(text, "", company_name)
                else:  # pp
                    _, _, summary_100, summary_25 = summarize_tos_pp("", text, company_name)
                
                # Check for summarization errors
                if summary_100.startswith("Error:") or summary_25.startswith("Error:"):
                    raise ValueError(f"Summarization failed: {summary_100}")
                
                # Step 5: Calculate simplified text mining metrics
                text_metrics = calculate_simple_metrics(text)
                
                # All steps succeeded - now save to database
                c.execute(
                    """INSERT OR IGNORE INTO scrapes (
                        domain, agreement_type, url, text_length, summary_100, summary_25,
                        text_metrics
                    ) VALUES (?,?,?,?,?,?,?)""",
                    (
                        d, agreement_type, url, len(text.split()), summary_100, summary_25,
                        str(text_metrics)
                    )
                )
                scrape_id = c.lastrowid
                conn.commit()
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Format processing time
                if processing_time < 1:
                    time_str = f"{processing_time*1000:.0f}ms"
                else:
                    time_str = f"{processing_time:.1f}s"
                
                # Add result to display list
                result = {
                    "domain": d,
                    "agreement_type": agreement_type,
                    "url": url,
                    "text_length": len(text.split()),
                    "summary_100": summary_100,
                    "summary_25": summary_25,
                    "text_metrics": text_metrics,
                    "processing_time": time_str,
                    "note": "Successfully processed and saved to database"
                }
                
                results.append(result)
                
            except Exception as e:
                # Any error in the process - don't save to database
                error_message = str(e)
                
                # Add specific user-friendly message for Cloudflare/CAPTCHA detection
                if "Cloudflare or CAPTCHA protection detected" in error_message:
                    flash(f"Error processing '{d}': Cloudflare or CAPTCHA protection detected. Cannot process this site.", "danger")
                else:
                    flash(f"Error processing '{d}': {error_message}", "danger")
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Format processing time
                if processing_time < 1:
                    time_str = f"{processing_time*1000:.0f}ms"
                else:
                    time_str = f"{processing_time:.1f}s"
                
                # Add error result to display list
                result = {
                    "domain": d,
                    "agreement_type": agreement_type,
                    "url": url if 'url' in locals() else "Not found",
                    "text_length": len(text.split()) if 'text' in locals() else 0,
                    "summary_100": f"Error: {error_message}",
                    "summary_25": f"Error: Processing failed",
                    "text_metrics": text_metrics if 'text_metrics' in locals() else calculate_simple_metrics(""),
                    "processing_time": time_str,
                    "note": f"Failed: {error_message} - Not saved to database"
                }
                
                results.append(result)
            
            # Be polite with the server
            time.sleep(1)

    # Render with the selected agreement type to maintain form state
    return render_template("index.html", results=results, selected_type=selected_type)

@app.route("/sql_viewer", methods=["GET"])
@login_required
def sql_viewer():
    # Create a new cursor for this function
    sql_cursor = conn.cursor()
    
    # Get list of tables
    sql_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = sql_cursor.fetchall()
    table_list = [table[0] for table in tables]
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    page_size = 10  # Reduced page size to 10 items per page
    offset = (page - 1) * page_size
    
    # Default to 'scrapes' table if present, else use the first available table
    default_table = 'scrapes' if 'scrapes' in table_list else (table_list[0] if table_list else None)
    selected_table = request.args.get('table', default_table)
    selected_columns = request.args.getlist('columns')
    
    # Get agreement_type filter value (for scrapes table)
    agreement_type_filter = request.args.get('agreement_type', '')
    
    table_data = None
    columns = None
    has_more = False
    total_rows = 0
    
    if selected_table:
        # Get column information
        sql_cursor.execute(f"PRAGMA table_info({selected_table})")
        columns = [column[1] for column in sql_cursor.fetchall()]
        
        # If specific columns are selected, use them
        if selected_columns:
            cols_str = ", ".join(selected_columns)
        else:
            cols_str = "*"
            selected_columns = columns  # Select all columns by default
        
        # Build the SQL query
        sql_query = f"SELECT {cols_str} FROM {selected_table}"
        count_query = f"SELECT COUNT(*) FROM {selected_table}"
        query_params = []
        
        # Add agreement_type filter for scrapes table
        if selected_table == 'scrapes' and agreement_type_filter:
            sql_query += " WHERE agreement_type = ?"
            count_query += " WHERE agreement_type = ?"
            query_params.append(agreement_type_filter)
            
        # Add pagination
        sql_query += " LIMIT ? OFFSET ?"
        query_params.extend([page_size, offset])
        
        # Get total row count (with filters applied)
        if selected_table == 'scrapes' and agreement_type_filter:
            sql_cursor.execute(count_query, [agreement_type_filter])
        else:
            sql_cursor.execute(count_query)
        total_rows = sql_cursor.fetchone()[0]
        
        # Get table data with pagination and filters
        sql_cursor.execute(sql_query, query_params)
        table_data = sql_cursor.fetchall()
        
        # Check if there are more rows
        has_more = offset + len(table_data) < total_rows
    
    # Close the cursor when done
    sql_cursor.close()
    
    return render_template("sql_viewer.html", 
                          tables=table_list, 
                          selected_table=selected_table,
                          all_columns=columns,
                          selected_columns=selected_columns,
                          table_data=table_data,
                          page=page,
                          total_rows=total_rows,
                          has_more=has_more,
                          total_pages=(total_rows // page_size) + (1 if total_rows % page_size > 0 else 0),
                          agreement_type_filter=agreement_type_filter)

@app.route("/api/process", methods=["POST"])
def api_process():
    """
    API endpoint to process a URL for TOS or PP analysis
    Expects JSON with:
    {
        "url": "example.com",
        "agreement_type": "tos" or "pp"
    }
    """
    # Get JSON data from request
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        # Extract domain and agreement type
        domain = data.get("url")
        agreement_type = data.get("agreement_type")
        
        # Validate inputs
        if not domain:
            return jsonify({"success": False, "error": "URL is required"}), 400
        
        if agreement_type not in ["tos", "pp"]:
            return jsonify({"success": False, "error": "Agreement type must be 'tos' or 'pp'"}), 400
        
        # Check if record already exists
        c.execute("SELECT id FROM scrapes WHERE domain = ? AND agreement_type = ?", (domain, agreement_type))
        existing_record = c.fetchone()
        
        if existing_record:
            # Record already exists, return it
            c.execute("SELECT * FROM scrapes WHERE id = ?", (existing_record[0],))
            record = c.fetchone()
            
            # Extract company name and prefix summaries
            company_name = extract_company_name(domain)
            
            # Format the result for API response
            text_metrics = eval(record[7])  # Convert the string back to a dictionary
            
            result = {
                "success": True,
                "domain": record[1],
                "agreement_type": record[2],
                "url": record[3],
                "text_length": record[4],
                "summary_100": record[5],
                "summary_25": record[6],
                "text_metrics": text_metrics,
                "message": "Retrieved from database (already exists)"
            }
            
            return jsonify(result)
        
        # All-or-nothing approach: only save to DB if everything succeeds
        try:
            start_time = time.time()  # Start timing
            
            # Step 1: Find policy links
            tos_url, pp_url = find_policy_links(domain)
            
            # Select URL based on agreement type
            url = tos_url if agreement_type == 'tos' else pp_url
            if not url:
                doc_type = 'Terms of Service' if agreement_type=='tos' else 'Privacy Policy'
                raise ValueError(f"No valid {doc_type} URL found for '{domain}'")
            
            # Step 2: Fetch and analyze text
            text = fetch_text(url)
            if text.startswith("https"):
                text = fetch_text(text)
            
            if not text:
                raise ValueError(f"No content found at {url}")
                
            # Step 2.5: Check if there are at least 200 words in the text
            word_count = len(text.split())
            if word_count < 200:
                raise ValueError(f"Text content too short ({word_count} words). Minimum 200 words required for processing.")
            
            # Step 3: Extract company name from domain
            company_name = extract_company_name(domain)
                
            # Step 4: Calculate summaries based on agreement type
            if agreement_type == "tos":
                summary_100, summary_25, _, _ = summarize_tos_pp(text, "", company_name)
            else:  # pp
                _, _, summary_100, summary_25 = summarize_tos_pp("", text, company_name)
            
            # Check for summarization errors
            if summary_100.startswith("Error:") or summary_25.startswith("Error:"):
                raise ValueError(f"Summarization failed: {summary_100}")
            
            # Step 5: Calculate simplified text mining metrics
            text_metrics = calculate_simple_metrics(text)
            
            # All steps succeeded - now save to database
            c.execute(
                """INSERT OR IGNORE INTO scrapes (
                    domain, agreement_type, url, text_length, summary_100, summary_25,
                    text_metrics
                ) VALUES (?,?,?,?,?,?,?)""",
                (
                    domain, agreement_type, url, len(text.split()), summary_100, summary_25,
                    str(text_metrics)
                )
            )
            scrape_id = c.lastrowid
            conn.commit()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            time_str = f"{processing_time:.2f}s"
            
            # Return success response
            result = {
                "success": True,
                "domain": domain,
                "agreement_type": agreement_type,
                "url": url,
                "text_length": len(text.split()),
                "summary_100": summary_100,
                "summary_25": summary_25,
                "text_metrics": text_metrics,
                "processing_time": time_str,
                "message": "Successfully processed and saved to database"
            }
            
            return jsonify(result)
            
        except Exception as e:
            # Any error in the process - don't save to database
            error_message = str(e)
            
            # Calculate processing time
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            time_str = f"{processing_time:.2f}s"
            
            # Special handling for Cloudflare/CAPTCHA detection
            if "Cloudflare or CAPTCHA protection detected" in error_message:
                # Return specific error code and message for Cloudflare/CAPTCHA detection
                result = {
                    "success": False,
                    "domain": domain,
                    "agreement_type": agreement_type,
                    "url": url if 'url' in locals() else "Not found",
                    "error": "Cloudflare or CAPTCHA protection detected. Cannot process this site.",
                    "error_details": error_message,
                    "processing_time": time_str,
                    "message": "Failed: Cloudflare or CAPTCHA protection detected - Not saved to database"
                }
                return jsonify(result), 403  # 403 Forbidden is appropriate for this case
            else:
                # Return error response for other errors
                result = {
                    "success": False,
                    "domain": domain,
                    "agreement_type": agreement_type,
                    "url": url if 'url' in locals() else "Not found",
                    "error": error_message,
                    "processing_time": time_str,
                    "message": f"Failed: {error_message} - Not saved to database"
                }
                
                return jsonify(result), 400
            
    except Exception as e:
        # Handle any unexpected exceptions
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Remove spaCy model loading
# try:
#     nlp = spacy.load("en_core_web_sm")
# except:
#     import subprocess
#     subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
#     nlp = spacy.load("en_core_web_sm")

if __name__=="__main__":
    app.run(debug=True)
