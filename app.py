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

# Create tables if they don't exist
c.execute("""
CREATE TABLE IF NOT EXISTS scrapes (
    id INTEGER PRIMARY KEY,
    domain TEXT,
    agreement_type TEXT,  -- 'tos' or 'pp'
    url TEXT,            -- URL of the document
    text TEXT,           -- Full text of the document
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
    raw_text TEXT,
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
                
                tos_summary_100 = (
                    "The Terms of Service govern the relationship between users and the service provider. "
                    "Based on the document's initial section: \"" + first_paragraph + "...\" "
                    "This legal agreement typically covers usage rights, content policies, account requirements, and liability limitations."
                )
                
                tos_summary_25 = "Terms outline usage rules, user obligations, and service provider rights regarding content and account access."
            except:
                tos_summary_100 = "The Terms of Service establish the legal agreement between users and the service provider. They typically cover acceptable use policies, intellectual property rights, account termination conditions, and liability limitations."
                tos_summary_25 = "Terms outline user obligations and service provider rights regarding platform usage."
        else:
            tos_summary_100 = "No terms of service document was found for this website."
            tos_summary_25 = "No Terms of Service document found."
            
        if pp_text:
            try:
                pp_sentences = sent_tokenize(pp_text)[:8]
                first_paragraph = " ".join(pp_sentences[:3])[:400]
                
                pp_summary_100 = (
                    "The Privacy Policy explains how the service handles user data. "
                    "Based on the document's initial section: \"" + first_paragraph + "...\" "
                    "This policy typically details what information is collected, how it's used, third-party sharing practices, and user privacy controls."
                )
                
                pp_summary_25 = "Policy describes data collection, usage, sharing practices, and user privacy options."
            except:
                pp_summary_100 = "The Privacy Policy outlines how user data is collected, processed, and shared."
                pp_summary_25 = "Policy explains data collection, usage, sharing practices, and privacy controls."
        else:
            pp_summary_100 = "No privacy policy document was found for this website."
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
                
                # Format the result to display
                text_metrics = eval(record[7])  # Convert the string back to a dictionary
                
                # Calculate processing time (minimal since we're just retrieving)
                processing_time = time.time() - start_time
                time_str = f"{processing_time*1000:.0f}ms"
                
                # Add retrieved result to display list
                result = {
                    "domain": record[1],
                    "agreement_type": record[2],
                    "url": record[3],
                    "text": record[4],
                    "summary_100": record[5],
                    "summary_25": record[6],
                    "text_metrics": text_metrics,
                    "processing_time": time_str,
                    "note": "Retrieved from database (already exists)"
                }
                
                results.append(result)
                continue
            
            # find policy links
            tos_url, pp_url = find_policy_links(d)
            
            # Select URL based on agreement type
            url = tos_url if agreement_type == 'tos' else pp_url
            if not url:
                doc_type = 'Terms of Service' if agreement_type=='tos' else 'Privacy Policy'
                flash(f"No valid {doc_type} URL found for '{d}'", "danger")
                continue
            
            # Fetch and analyze text
            text = fetch_text(url)
            if text.startswith("https"):
                text = fetch_text(text)
            
            if not text:
                flash(f"No content found at {url}", "danger")
                continue
            
            # Calculate summaries based on agreement type
            if agreement_type == "tos":
                raw_output, summary_100, summary_25, _, _ = summarize_tos_pp(text, "")
            else:  # pp
                raw_output, _, _, summary_100, summary_25 = summarize_tos_pp("", text)
            
            # Calculate simplified text mining metrics
            text_metrics = calculate_simple_metrics(text)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Save to database
            c.execute(
                """INSERT OR IGNORE INTO scrapes (
                    domain, agreement_type, url, text, summary_100, summary_25,
                    text_metrics
                ) VALUES (?,?,?,?,?,?,?)""",
                (
                    d, agreement_type, url, text, summary_100, summary_25,
                    str(text_metrics)
                )
            )
            scrape_id = c.lastrowid
            conn.commit()
            
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
                "text": text,
                "summary_100": summary_100,
                "summary_25": summary_25,
                "text_metrics": text_metrics,
                "processing_time": time_str
            }
            
            results.append(result)
            time.sleep(1)  # be polite

    # Render with the selected agreement type to maintain form state
    return render_template("index.html", results=results, selected_type=selected_type)

@app.route("/sql_viewer", methods=["GET"])
@login_required
def sql_viewer():
    # Get list of tables
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = c.fetchall()
    table_list = [table[0] for table in tables]
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    page_size = 100
    offset = (page - 1) * page_size
    
    # Default to 'scrapes' table if present, else use the first available table
    default_table = 'scrapes' if 'scrapes' in table_list else (table_list[0] if table_list else None)
    selected_table = request.args.get('table', default_table)
    selected_columns = request.args.getlist('columns')
    
    table_data = None
    columns = None
    has_more = False
    total_rows = 0
    
    if selected_table:
        # Get column information
        c.execute(f"PRAGMA table_info({selected_table})")
        columns = [column[1] for column in c.fetchall()]
        
        # Get total row count
        c.execute(f"SELECT COUNT(*) FROM {selected_table}")
        total_rows = c.fetchone()[0]
        
        # If specific columns are selected, use them
        if selected_columns:
            cols_str = ", ".join(selected_columns)
        else:
            cols_str = "*"
            selected_columns = columns  # Select all columns by default
        
        # Get table data with pagination
        c.execute(f"SELECT {cols_str} FROM {selected_table} LIMIT {page_size} OFFSET {offset}")
        table_data = c.fetchall()
        
        # Check if there are more rows
        has_more = offset + len(table_data) < total_rows
    
    return render_template("sql_viewer.html", 
                          tables=table_list, 
                          selected_table=selected_table,
                          all_columns=columns,
                          selected_columns=selected_columns,
                          table_data=table_data,
                          page=page,
                          total_rows=total_rows,
                          has_more=has_more,
                          total_pages=(total_rows // page_size) + (1 if total_rows % page_size > 0 else 0))

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
