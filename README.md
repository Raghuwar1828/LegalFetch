# LegalFetch

## Project Overview
LegalFetch is a crawler-based software tool designed to extract Terms of Service (TOS) and Privacy Policy (PP) content from publicly available web pages. It analyzes the text and provides insights such as summaries, word frequency statistics, and key metrics.

## Features
- *Web Scraping* – Extract TOS/PP from publicly available sources.
- *Data Storage* – Store extracted data in a structured MySQL database.
- *Text Processing* – Perform summarization (100-word & 1-sentence), word frequency analysis, and other text mining metrics.
- *Search & Filtering* – Allow users to query and analyze extracted TOS/PP.
- *User Interface* – Web-based UI for easy access to results.
- *Scalability* – Handle at least 1 million TOS & PP entries.

## Tech Stack

### *Frontend (User Interface)*
- *React.js* – Build a responsive & interactive UI.
- *HTML, CSS, JavaScript* – Basic structure & styling.

### *Backend (API & Logic)*
- *PHP (Laravel/Core PHP)* – API development & backend logic.
- *MySQL/PostgreSQL* – Database to store TOS & PP text.
- *Scrapy / Apache Nutch / CRWLR* – Web crawling for extracting text.

### *Hosting & Deployment*
- *Hostinger* – Hosting the web application.
- *cPanel & FTP* – Managing deployment on Hostinger.
- *Apache / Nginx* – Web server to host APIs.

## Step-by-Step Process

### *User Input (Frontend)*
1. Users enter a website URL to fetch TOS/PP.
2. UI displays results after processing.

### *Web Crawling (Backend)*
1. Scrapy / Apache Nutch fetches TOS/PP from URLs.
2. Stores raw text in the database.

### *Text Processing (Backend)*
PHP functions for:
- Summarization (100-word & 1-sentence).
- Word frequency analysis.
- Other text mining techniques.

### *Data Storage (Database)*
Stores:
- Website URL.
- Extracted TOS/PP text.
- Processed summary & analysis results.

### *Data Display (Frontend)*
- Results displayed using React.js.

## Deployment Plan
1. Set up Hostinger hosting.
2. Upload React frontend & PHP backend.
3. Deploy MySQL database & connect APIs.
4. Schedule scrapers (Cron Jobs for periodic crawling).