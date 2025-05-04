# LegalFetch with Gemini API Integration

This application uses the Google Gemini API to analyze Terms of Service and Privacy Policy documents.

## Setup Instructions

1. Get a Gemini API key from [Google AI Studio](https://ai.google.dev/).

2. Create a `.env` file in the root directory with your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

## Testing the Gemini API

You can test if your Gemini API is working correctly by running:

```
python test_gemini_http.py
```

This script will make a simple request to the Gemini API and display the response.

## API Information

This application uses the Gemini 2.0 Flash-Lite model, which provides:
- Cost-efficient performance
- Multimodal input support (text, image, video, audio)
- 1 million token context window
- Maximum 8k token output

## Troubleshooting

If you encounter issues with the API:

1. Verify your API key is correct
2. Check that you have internet connectivity
3. Look for error messages in the console output
4. Try running the test script to isolate API-specific issues

## LegalFetch

LegalFetch is a web application that analyzes Terms of Service and Privacy Policy documents.

## Features

- Analyzes and summarizes Terms of Service and Privacy Policy documents
- Extracts key metrics like readability, sentiment, and word frequency
- Provides API access for document processing

## Majestic Million Scraper

The application includes a utility script to scrape TOS/PP documents from top websites using the Majestic Million dataset.

### Usage

1. Make sure the Flask application is running locally
2. Run the scraper script:

```bash
python scrape_majestic.py
```

3. Follow the prompts to:
   - Select whether to scrape Terms of Service (TOS), Privacy Policy (PP), or both from the numbered menu
   - Set a success target (number of successfully processed websites to achieve)
   - Continue from where you left off in a previous session or start from rank 1 by default
   - Enter a starting rank (1-1,000,000) in the Majestic Million list

### Progress Tracking

The script provides detailed progress tracking with:

- Current rank being processed
- Number of successful scrapes so far
- Remaining scrapes needed to reach your target
- Percentage completion toward your target

After each website is processed, you'll see a progress report showing this information. The script will automatically continue until your success target is reached.

### Processing Options

The script offers three processing modes:

- **TOS**: Processes only Terms of Service documents for each website
- **PP**: Processes only Privacy Policy documents for each website
- **Both**: Processes both TOS and PP for each website in sequence

When using the "both" option, the script will attempt to find and process both document types for each website. Results will be classified as:
- Full success: Both TOS and PP processed successfully
- Partial success: Only one document type was processed successfully
- Failure: Neither document type was processed successfully

The script will:
- Download the Majestic Million dataset if not already present
- Process websites starting from your specified rank
- Send each website to your local API endpoint
- Save successfully processed documents to your database
- Track your progress in a file called `scraper_progress.json` for easy resuming later

### Resuming a Previous Session

The scraper saves your progress after each processed website. When you restart the script:

1. Select the same agreement type (tos/pp) you were working with
2. The script will detect your previous progress and offer to continue from the next rank
3. If you accept, it will resume processing from the next website in the ranked list

## API Endpoint

The application provides a REST API endpoint for processing documents:

```bash
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{"url": "example.com", "agreement_type": "tos"}'
```

Response example (success):
```json
{
  "success": true,
  "domain": "example.com",
  "url": "https://example.com/terms",
  "agreement_type": "tos",
  "summary_25": "Brief summary...",
  "summary_100": "Longer summary...",
  "text_metrics": { ... },
  "processing_time": "2.91s",
  "message": "Successfully processed and saved to database"
}
``` 