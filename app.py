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
    session, flash

)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI
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

# Access your OpenAI API key from the environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    print("WARNING: OPENAI_API_KEY not found in environment variables. Summarization will use local fallback methods.")

# NVIDIA API endpoint configuration
nvidia_api_base = os.environ.get("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")

txt = ""
# --- CONFIGURATION ---
app = Flask(__name__)
app.secret_key = "YOUR_SECRET_KEY_HERE"  # ← change this!

# Initialize NVIDIA/OpenAI client
# Initialize NVIDIA Ollama-like OpenAI Endpoint
client = OpenAI(
    base_url=nvidia_api_base,
    api_key=openai_api_key
)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9"
}

# --- DATABASE SETUP ---
conn = sqlite3.connect("app.db", check_same_thread=False)
c = conn.cursor()
# users table
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    password TEXT
)
""")
# scrapes table
c.execute("""
CREATE TABLE IF NOT EXISTS scrapes (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    domain TEXT,
    tos_url TEXT,
    pp_url TEXT,
    tos_text TEXT,
    pp_text TEXT,
    tos_summary_100 TEXT,
    tos_summary_25 TEXT,
    pp_summary_100 TEXT,
    pp_summary_25 TEXT,
    freq_tos TEXT,
    freq_pp TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
""")
# summary cache table
c.execute("""
CREATE TABLE IF NOT EXISTS summary_cache (
    id INTEGER PRIMARY KEY,
    tos_hash INTEGER,
    pp_hash INTEGER,
    raw_text TEXT,
    tos_summary_100 TEXT,
    tos_summary_25 TEXT,
    pp_summary_100 TEXT,
    pp_summary_25 TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
# Add an index for faster lookups
c.execute("""
CREATE INDEX IF NOT EXISTS idx_summary_cache 
ON summary_cache(tos_hash, pp_hash)
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

        tos_link, pp_link = "", ""
        for link in links:
            href = link["href"].lower()
            if 'terms' in href and not tos_link:
                tos_link = get_full_url("https://" + base_url, href)
            if 'privacy' in href and not pp_link:
                pp_link = get_full_url("https://" + base_url, href)

        return tos_link, pp_link
    except Exception as e:
        print(f"[!] Error fetching {base_url}: {e}")
        return "", ""

def fetch_text(url):
    """ Fetch all <p> text from url """
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        print(f"[!] Error scraping {url}: {e}")
        return ""

def clean_tokens(text):
    stops = set(stopwords.words('english'))
    t = re.sub(r'[^a-z\s]', '', text.lower())
    return [w for w in t.split() if w and w not in stops]

def summarize_tos_pp(tos_text, pp_text):
    # First check if we have a cached result for this exact content
    tos_hash = hash(tos_text[:1000] if tos_text else "")  # Use first 1000 chars as hash key
    pp_hash = hash(pp_text[:1000] if pp_text else "")
    
    # Check if we have cached results in the database
    c.execute("""
        SELECT raw_text, tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25 
        FROM summary_cache 
        WHERE tos_hash = ? AND pp_hash = ?
    """, (tos_hash, pp_hash))
    
    cached = c.fetchone()
    if cached:
        return cached[0], cached[1], cached[2], cached[3], cached[4]
    
    # If no content to summarize, return empty strings
    if not tos_text and not pp_text:
        return "", "No terms of service found", "No TOS found", "No privacy policy found", "No PP found"
    
    prompt = (
        "You are a legal document summarizer. Generate clear, accurate, and meaningful summaries of legal documents in a specific format.\n\n"
        "Instructions:\n"
        "1. Carefully analyze the provided Terms of Service (TOS) and Privacy Policy (PP) texts\n"
        "2. Create summaries that include direct quotes from the document\n"
        "3. Use the exact format shown below for each summary\n"
        "4. Focus on the most important legal implications for users\n\n"
        
        "TOS detailed summary (90-110 words): Start with 'The Terms of Service govern the relationship between users and the service provider. " 
        "Based on the document's initial section: \"[INSERT DIRECT QUOTE FROM DOCUMENT HERE]\" This legal agreement typically covers [KEY ASPECTS BASED ON ANALYSIS].' " 
        "Include an actual quote from the document between the quotation marks.\n\n"
        
        "TOS short summary (15-35 words): Create a concise summary focusing on the most critical user obligations and rights.\n\n"
        
        "PP detailed summary (90-110 words): Start with 'The Privacy Policy explains how the service handles user data. "
        "Based on the document's initial section: \"[INSERT DIRECT QUOTE FROM DOCUMENT HERE]\" This policy covers [KEY ASPECTS BASED ON ANALYSIS].' "
        "Include an actual quote from the document between the quotation marks.\n\n"
        
        "PP short summary (15-35 words): Create a concise summary focusing on data collection and privacy implications.\n\n"
        
        f"TOS Text: {tos_text[:5000]}\n\nPP Text: {pp_text[:5000]}"
    )

    try:
        # Check if API key exists
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
            
        # Use a non-streaming approach first as fallback
        try:
            # Try non-streaming first for reliability
            resp = client.chat.completions.create(
                model="nvidia/llama-3.3-nemotron-base-8b",
                messages=[{"role":"system","content": prompt}],
                temperature=0.6,
                top_p=0.95,
                max_tokens=1000,
                frequency_penalty=0,
                presence_penalty=0
            )
            full_txt = resp.choices[0].message.content
        except Exception as stream_error:
            print(f"Non-streaming API failed: {stream_error}")
            
            # Try streaming as fallback
            resp = client.chat.completions.create(
                model="nvidia/llama-3.3-nemotron-base-8b",
                messages=[{"role":"system","content": prompt}],
                temperature=0.6,
                top_p=0.95,
                max_tokens=1000,
                frequency_penalty=0,
                presence_penalty=0,
                stream=True
            )
            
            # Accumulate chunks
            full_txt = ""
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_txt += delta
      
        # Split into clean lines
        lines = [line.strip() for line in full_txt.splitlines() if line.strip()]
        
        # Extract summaries
        tos_summary_100 = ""
        tos_summary_25 = ""
        pp_summary_100 = ""
        pp_summary_25 = ""
        
        for idx, line in enumerate(lines):
            low = line.lower()
            
            # TOS summaries
            if "tos detailed summary" in low or "tos 100 word summary" in low:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    tos_summary_100 = parts[1].strip()
                elif idx+1 < len(lines):
                    tos_summary_100 = lines[idx+1].lstrip("- ").strip()
                    
            if "tos short summary" in low or "tos 25 word summary" in low:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    tos_summary_25 = parts[1].strip()
                elif idx+1 < len(lines):
                    tos_summary_25 = lines[idx+1].lstrip("- ").strip()
            
            # PP summaries        
            if "pp detailed summary" in low or "pp 100 word summary" in low:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    pp_summary_100 = parts[1].strip()
                elif idx+1 < len(lines):
                    pp_summary_100 = lines[idx+1].lstrip("- ").strip()
                    
            if "pp short summary" in low or "pp 25 word summary" in low:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    pp_summary_25 = parts[1].strip()
                elif idx+1 < len(lines):
                    pp_summary_25 = lines[idx+1].lstrip("- ").strip()

        # Verify and adjust summary lengths
        def adjust_summary(summary, target_length, min_length, max_length, fallback):
            if not summary:
                return fallback
                
            word_count = len(summary.split())
            
            if word_count < min_length:
                summary += f" (Note: This summary contains only {word_count} words, below the target of {target_length} words)"
            elif word_count > max_length:
                # Truncate to max_length words
                words = summary.split()
                summary = " ".join(words[:max_length]) + "..."
                
            return summary
        
        # Apply length checks and adjustments
        tos_summary_100 = adjust_summary(
            tos_summary_100, 100, 90, 110, 
            "The terms of service outline the user's rights and responsibilities when using the service. They typically cover acceptable use policies, content ownership, account termination conditions, and liability limitations. Users must agree to these terms to access the service, and violations may result in account suspension or termination. The service provider usually reserves the right to modify these terms, with continued use signifying acceptance of changes."
        )
        
        tos_summary_25 = adjust_summary(
            tos_summary_25, 25, 15, 35,
            "Terms govern user rights, responsibilities, and restrictions while using the service."
        )
        
        pp_summary_100 = adjust_summary(
            pp_summary_100, 100, 90, 110,
            "The privacy policy explains how user data is collected, processed, stored, and shared. It typically details what personal information is gathered, how cookies and tracking technologies are implemented, third-party data sharing practices, and security measures in place. Users may have options to control certain data collection aspects, though some information is essential for service functionality. The policy also outlines user rights regarding their personal information."
        )
        
        pp_summary_25 = adjust_summary(
            pp_summary_25, 25, 15, 35,
            "Policy explains how user data is collected, used, stored, and shared with third parties."
        )

        # Cache the results
        c.execute("""
            INSERT INTO summary_cache 
            (tos_hash, pp_hash, raw_text, tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (tos_hash, pp_hash, full_txt, tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25))
        
        conn.commit()
        
        return full_txt, tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25
        
    except Exception as e:
        print(f"Error in summarization: {e}")
        
        # Use local fallback summaries if API fails
        if tos_text:
            try:
                # Create a meaningful paragraph from first few sentences
                tos_sentences = sent_tokenize(tos_text)[:8]
                first_paragraph = " ".join(tos_sentences[:3])[:400]
                
                # Create a more structured manual summary
                tos_summary_100 = (
                    "The Terms of Service govern the relationship between users and the service provider. "
                    "Based on the document's initial section: \"" + first_paragraph + "...\" "
                    "This legal agreement typically covers usage rights, content policies, account requirements, and liability limitations."
                )
                
                tos_summary_25 = "Terms outline usage rules, user obligations, and service provider rights regarding content and account access."
            except:
                # Ultimate fallback
                tos_summary_100 = "The Terms of Service establish the legal agreement between users and the service provider. They typically cover acceptable use policies, intellectual property rights, account termination conditions, and liability limitations. Users must comply with these terms to maintain service access."
                tos_summary_25 = "Terms outline user obligations and service provider rights regarding platform usage."
        else:
            tos_summary_100 = "No terms of service document was found for this website. Without Terms of Service, it's unclear what rules govern the use of this service and what rights users have when interacting with it."
            tos_summary_25 = "No Terms of Service document found."
            
        if pp_text:
            try:
                # Create a meaningful paragraph from first few sentences
                pp_sentences = sent_tokenize(pp_text)[:8]
                first_paragraph = " ".join(pp_sentences[:3])[:400]
                
                # Create a more structured manual summary
                pp_summary_100 = (
                    "The Privacy Policy explains how the service handles user data. "
                    "Based on the document's initial section: \"" + first_paragraph + "...\" "
                    "This policy typically details what information is collected, how it's used, third-party sharing practices, and user privacy controls."
                )
                
                pp_summary_25 = "Policy describes data collection, usage, sharing practices, and user privacy options."
            except:
                # Ultimate fallback
                pp_summary_100 = "The Privacy Policy outlines how user data is collected, processed, and shared. It typically covers what personal information is gathered, cookie usage, third-party data sharing, security measures, and user rights regarding their information."
                pp_summary_25 = "Policy explains data collection, usage, sharing practices, and privacy controls."
        else:
            pp_summary_100 = "No privacy policy document was found for this website. Without a Privacy Policy, it's unclear how this service handles user data, what information they collect, or how they protect user privacy."
            pp_summary_25 = "No Privacy Policy document found."
            
        return "", tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25



def analyze_tos_pp(tos_text, pp_text):
    # clean & tokens
    tok_tos = clean_tokens(tos_text)
    tok_pp  = clean_tokens(pp_text)
    freq_tos = Counter(tok_tos).most_common(10)
    freq_pp  = Counter(tok_pp).most_common(10)

    raw_output, tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25 = summarize_tos_pp(tos_text, pp_text)
    return raw_output, freq_tos, freq_pp, tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25

def calculate_text_metrics(tos_text, pp_text):
    metrics = {}
    
    # 1. TF-IDF Analysis
    documents = [tos_text, pp_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    # Top TF-IDF words for TOS
    tos_tfidf = tfidf_matrix[0].toarray()[0]
    tos_top_indices = np.argsort(tos_tfidf)[::-1][:10]
    tos_top_tfidf = [(feature_names[i], round(tos_tfidf[i], 3)) for i in tos_top_indices]
    
    # Top TF-IDF words for PP
    pp_tfidf = tfidf_matrix[1].toarray()[0] if len(documents) > 1 else []
    pp_top_indices = np.argsort(pp_tfidf)[::-1][:10] if len(documents) > 1 else []
    pp_top_tfidf = [(feature_names[i], round(pp_tfidf[i], 3)) for i in pp_top_indices] if len(documents) > 1 else []
    
    metrics['tfidf_tos'] = tos_top_tfidf
    metrics['tfidf_pp'] = pp_top_tfidf
    
    # 2. Document Length Metrics
    # For TOS
    if tos_text:
        tos_words = word_tokenize(tos_text)
        tos_sentences = sent_tokenize(tos_text)
        tos_num_words = len(tos_words)
        tos_num_sentences = len(tos_sentences)
        tos_avg_sentence_length = tos_num_words / tos_num_sentences if tos_num_sentences > 0 else 0
        
        metrics['doc_length_tos'] = {
            'num_words': tos_num_words,
            'num_sentences': tos_num_sentences,
            'avg_sentence_length': round(tos_avg_sentence_length, 2)
        }
    else:
        metrics['doc_length_tos'] = {
            'num_words': 0,
            'num_sentences': 0,
            'avg_sentence_length': 0
        }
        
    # For PP
    if pp_text:
        pp_words = word_tokenize(pp_text)
        pp_sentences = sent_tokenize(pp_text)
        pp_num_words = len(pp_words)
        pp_num_sentences = len(pp_sentences)
        pp_avg_sentence_length = pp_num_words / pp_num_sentences if pp_num_sentences > 0 else 0
        
        metrics['doc_length_pp'] = {
            'num_words': pp_num_words,
            'num_sentences': pp_num_sentences,
            'avg_sentence_length': round(pp_avg_sentence_length, 2)
        }
    else:
        metrics['doc_length_pp'] = {
            'num_words': 0,
            'num_sentences': 0,
            'avg_sentence_length': 0
        }
    
    # 3. Hapax Legomena Ratio (Vocabulary Richness)
    # For TOS
    if tos_text:
        tos_words = word_tokenize(tos_text.lower())
        tos_word_counts = Counter(tos_words)
        tos_hapax_legomena = [word for word, count in tos_word_counts.items() if count == 1]
        tos_hapax_ratio = len(tos_hapax_legomena) / len(tos_words) if tos_words else 0
        
        metrics['hapax_tos'] = {
            'ratio': round(tos_hapax_ratio, 4),
            'count': len(tos_hapax_legomena),
            'examples': tos_hapax_legomena[:10]  # First 10 examples of words that appear only once
        }
    else:
        metrics['hapax_tos'] = {
            'ratio': 0,
            'count': 0,
            'examples': []
        }
        
    # For PP
    if pp_text:
        pp_words = word_tokenize(pp_text.lower())
        pp_word_counts = Counter(pp_words)
        pp_hapax_legomena = [word for word, count in pp_word_counts.items() if count == 1]
        pp_hapax_ratio = len(pp_hapax_legomena) / len(pp_words) if pp_words else 0
        
        metrics['hapax_pp'] = {
            'ratio': round(pp_hapax_ratio, 4),
            'count': len(pp_hapax_legomena),
            'examples': pp_hapax_legomena[:10]  # First 10 examples of words that appear only once
        }
    else:
        metrics['hapax_pp'] = {
            'ratio': 0,
            'count': 0,
            'examples': []
        }
    
    # 4. Readability Score
    metrics['readability_tos'] = textstat.flesch_reading_ease(tos_text) if tos_text else 0
    metrics['readability_pp'] = textstat.flesch_reading_ease(pp_text) if pp_text else 0
    
    # 5. Sentiment Analysis
    if tos_text:
        tos_blob = TextBlob(tos_text)
        metrics['sentiment_tos'] = round(tos_blob.sentiment.polarity, 3)
    else:
        metrics['sentiment_tos'] = 0
    
    if pp_text:
        pp_blob = TextBlob(pp_text)
        metrics['sentiment_pp'] = round(pp_blob.sentiment.polarity, 3)
    else:
        metrics['sentiment_pp'] = 0
    
    return metrics

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
@login_required
def index():
    results = []
    if request.method=="POST":
        doms = request.form["domains"]
        doc_type = request.form.get("doc_type", "tos")  # Default to TOS if not specified
        
        for d in [x.strip() for x in doms.split(",") if x.strip()]:
            # Check for existing scrape record to avoid duplicates
            c.execute(
                """
                SELECT tos_url, pp_url, tos_text, pp_text, tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25
                FROM scrapes
                WHERE user_id = ? AND domain = ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (session["user_id"], d)
            )
            cached = c.fetchone()
            if cached:
                # Use cached results
                tos_url, pp_url, tos_text, pp_text, tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25 = cached
                # Compute frequency and metrics based on selected document type
                if doc_type == "tos":
                    freq_tos = Counter(clean_tokens(tos_text)).most_common(10) if tos_text else []
                    freq_pp = []
                else:
                    freq_tos = []
                    freq_pp = Counter(clean_tokens(pp_text)).most_common(10) if pp_text else []
                text_metrics = calculate_text_metrics(tos_text, pp_text)
                # Append cached result
                results.append({
                    "domain": d,
                    "tos_url": tos_url,
                    "pp_url": pp_url,
                    "tos_text": tos_text,
                    "pp_text": pp_text,
                    "tos_summary_100": tos_summary_100,
                    "tos_summary_25": tos_summary_25,
                    "pp_summary_100": pp_summary_100,
                    "pp_summary_25": pp_summary_25,
                    "freq_tos": freq_tos,
                    "freq_pp": freq_pp,
                    "doc_type": doc_type,
                    # Add text mining metrics
                    "tfidf_tos": text_metrics['tfidf_tos'],
                    "tfidf_pp": text_metrics['tfidf_pp'],
                    "doc_length_tos": text_metrics['doc_length_tos'],
                    "doc_length_pp": text_metrics['doc_length_pp'],
                    "hapax_tos": text_metrics['hapax_tos'],
                    "hapax_pp": text_metrics['hapax_pp'],
                    "readability_tos": text_metrics['readability_tos'],
                    "readability_pp": text_metrics['readability_pp'],
                    "sentiment_tos": text_metrics['sentiment_tos'],
                    "sentiment_pp": text_metrics['sentiment_pp'],
                    "processing_time": "from cache"
                })
                continue  # Skip scraping and saving

            # No cached record, proceed with scraping
            start_time = time.time()
            tos_url, pp_url = find_policy_links(d)
            
            # Based on selected document type, only fetch and analyze that document
            tos_text = ""
            pp_text = ""
            
            if doc_type == "tos":
                tos_text = fetch_text(tos_url) if tos_url else ""
                if tos_text.startswith("https"):
                    tos_text = fetch_text(tos_text)
                freq_tos = Counter(clean_tokens(tos_text)).most_common(10) if tos_text else []
                freq_pp = []
            else:  # doc_type == "pp"
                pp_text = fetch_text(pp_url) if pp_url else ""
                if pp_text.startswith("https"):
                    pp_text = fetch_text(pp_text)
                freq_tos = []
                freq_pp = Counter(clean_tokens(pp_text)).most_common(10) if pp_text else []
            
            # Only call summarize_tos_pp once and use its returned values
            raw_output, tos_summary_100, tos_summary_25, pp_summary_100, pp_summary_25 = summarize_tos_pp(tos_text, pp_text)
            raw_html = markdown.markdown(raw_output, extensions=["fenced_code", "tables"])
            
            # Calculate text mining metrics
            text_metrics = calculate_text_metrics(tos_text, pp_text)
            
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Processing time for {d}: {processing_time:.2f} seconds")
            
            # save to DB
            c.execute("""
            INSERT INTO scrapes
            (user_id,domain,tos_url,pp_url,tos_text,pp_text,tos_summary_100,tos_summary_25,pp_summary_100,pp_summary_25,freq_tos,freq_pp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,(
                session["user_id"], d, tos_url, pp_url,
                tos_text, pp_text, tos_summary_100, tos_summary_25,
                pp_summary_100, pp_summary_25,
                str(freq_tos), str(freq_pp)
            ))
            conn.commit()

            results.append({
                "domain": d,
                "tos_url": tos_url,
                "pp_url": pp_url,
                "tos_text": tos_text,
                "pp_text": pp_text,
                "tos_summary_100": tos_summary_100,
                "tos_summary_25": tos_summary_25,
                "pp_summary_100": pp_summary_100,
                "pp_summary_25": pp_summary_25,
                "raw_output": raw_output,
                "raw_html": raw_html,
                "freq_tos": freq_tos,
                "freq_pp": freq_pp,
                "doc_type": doc_type,  # Add document type to the results
                # Add text mining metrics
                "tfidf_tos": text_metrics['tfidf_tos'],
                "tfidf_pp": text_metrics['tfidf_pp'],
                "doc_length_tos": text_metrics['doc_length_tos'],
                "doc_length_pp": text_metrics['doc_length_pp'],
                "hapax_tos": text_metrics['hapax_tos'],
                "hapax_pp": text_metrics['hapax_pp'],
                "readability_tos": text_metrics['readability_tos'],
                "readability_pp": text_metrics['readability_pp'],
                "sentiment_tos": text_metrics['sentiment_tos'],
                "sentiment_pp": text_metrics['sentiment_pp'],
                "processing_time": f"{processing_time:.2f}s"
            })
            time.sleep(1)  # be polite

    return render_template("index.html", results=results)

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
