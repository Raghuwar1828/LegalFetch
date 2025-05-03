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