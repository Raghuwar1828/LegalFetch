# Legal Fetch

A web application for fetching, analyzing, and summarizing legal documents like Terms of Service and Privacy Policies from websites.

## Features

- User authentication system for secure access
- Fetch terms of service and privacy policies from websites
- Analyze text with NLP techniques
- Generate word clouds and frequency analysis
- Summarize documents using AI
- Save and organize your analyses

## Technologies Used

- Python 3.10+
- Flask web framework
- NLTK for natural language processing
- OpenAI API for AI-powered summaries
- SQLite for database
- HTML/CSS for frontend

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Raghuwar1828/LegalFetch.git
cd LegalFetch
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key in a .env file:
```
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

1. Register an account or log in
2. Enter domain names in the search field
3. View and analyze the fetched legal documents
4. Save analyses for future reference

## License

This project is licensed under the MIT License - see the LICENSE file for details. 