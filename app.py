import sqlite3
import re
import time
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
from collections import Counter
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI
from wordcloud import WordCloud
import io, base64
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access your OpenAI API key from the environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

txt = ""
# --- CONFIGURATION ---
app = Flask(__name__)
app.secret_key = "YOUR_SECRET_KEY_HERE"  # ← change this!

# Initialize NVIDIA/OpenAI client
# Initialize NVIDIA Ollama-like OpenAI Endpoint
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
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
    summary_100 TEXT,
    summary_10 TEXT,
    freq_tos TEXT,
    freq_pp TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
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


def summarize_tos_pp(tos_text, pp_text):
    prompt = (
        "summarize both in 100 words and 10 words and give output like\n"
        "100 word summary : - <your 100-word summary>\n"
        "10 word summary : - <your 10-word summary>\n\n"
        f"TOS Text: {tos_text}\n\nPP Text: {pp_text}"
    )

    # Stream the response
    resp = client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        messages=[{"role":"system","content": prompt}],
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
        frequency_penalty=0,
        presence_penalty=0,
        stream=True
    )

    # 1) Accumulate every chunk
    full_txt = ""
    for chunk in resp:
        delta = chunk.choices[0].delta.content
        if delta:
            full_txt += delta
  
    # 2) Split into clean lines
    lines = [line.strip() for line in full_txt.splitlines() if line.strip()]
    print("hereeeeeee")
    print("lines", lines)
    # 3) Extract summaries
    summary_100 = ""
    summary_10  = ""
    for idx, line in enumerate(lines):
        low = line.lower()
        # when we see the header, grab the next bullet or inline text
        if "100 word summary" in low:
            # next line bullet?
            if idx+1 < len(lines) and lines[idx+1].startswith("-"):
                summary_100 = lines[idx+1].lstrip("- ").strip()
            else:
                # inline after colon?
                parts = line.split(":", 1)
                if len(parts) > 1:
                    summary_100 = parts[1].strip()
        if "10 word summary" in low:
            if idx+1 < len(lines) and lines[idx+1].startswith("-"):
                summary_10 = lines[idx+1].lstrip("- ").strip()
            else:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    summary_10 = parts[1].strip()

    return full_txt, summary_100, summary_10



def analyze_tos_pp(tos_text, pp_text):
    # clean & tokens
    stops = set(stopwords.words('english'))
    def clean_tokens(text):
        t = re.sub(r'[^a-z\s]', '', text.lower())
        return [w for w in t.split() if w and w not in stops]

    tok_tos = clean_tokens(tos_text)
    tok_pp  = clean_tokens(pp_text)
    freq_tos = Counter(tok_tos).most_common(10)
    freq_pp  = Counter(tok_pp).most_common(10)

    raw_text, sum100, sum10 = summarize_tos_pp(tos_text, pp_text)
    return raw_text, freq_tos, freq_pp, sum100, sum10
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
        for d in [x.strip() for x in doms.split(",") if x.strip()]:
            tos_url, pp_url = find_policy_links(d)
            tos_text = fetch_text(tos_url) if tos_url else ""
            pp_text  = fetch_text(pp_url)  if pp_url  else ""
            if tos_text.startswith("https") and pp_text.startswith("https"):
                tos_text = fetch_text(tos_text)
                pp_text = fetch_text(pp_text)
            raw_text, freq_tos, freq_pp, sum100, sum10 = analyze_tos_pp(tos_text, pp_text)
            raw_output, sum100, sum10 = summarize_tos_pp(tos_text, pp_text)
            raw_html = markdown.markdown(raw_output, extensions=["fenced_code", "tables"])
            combined = " ".join(tos_text + pp_text)

            # generate a wordcloud image
            wc = WordCloud(width=800, height=400, background_color="white").generate(combined)

            # dump to PNG in memory
            buf = io.BytesIO()
            wc.to_image().save(buf, format="PNG")
            buf.seek(0)

            # encode to base64 for embedding
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            wordcloud_data = f"data:image/png;base64,{img_b64}"

            # save to DB
            c.execute("""
            INSERT INTO scrapes
            (user_id,domain,tos_url,pp_url,tos_text,pp_text,summary_100,summary_10,freq_tos,freq_pp)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,(
                session["user_id"], d, tos_url, pp_url,
                tos_text, pp_text, sum100, sum10,
                str(freq_tos), str(freq_pp)
            ))
            conn.commit()

            results.append({
                "domain": d,
                "tos_url": tos_url,
                "pp_url": pp_url,
                "tos_text": tos_text,
                "pp_text": pp_text,
                "summary_100": sum100,
                "summary_10": sum10,
                "raw_output": raw_output,
                    "raw_html": raw_html,
                        "wordcloud": wordcloud_data,
                "freq_tos": freq_tos,
                "freq_pp": freq_pp
            })
            time.sleep(1)  # be polite

    return render_template("index.html", results=results)

if __name__=="__main__":
    app.run(debug=True)
