"""
Summarization Agent using FastAPI, Firecrawl, and OpenAI

Dependencies:
  - fastapi
  - uvicorn
  - openai>=1.63.0
  - firecrawl-py>=0.1.0
  - python-dotenv>=1.0.0
  - pydantic>=2.0.0

Usage (locally):
  uvicorn summarization_agent:app --reload

Environment:
  Set FIRECRAWL_API_KEY and OPENAI_API_KEY in your environment or a .env file.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import openai
from firecrawl import FirecrawlApp

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Summarization Agent")

# Get API keys from environment variables
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not FIRECRAWL_API_KEY or not OPENAI_API_KEY:
    sys.exit("Both FIRECRAWL_API_KEY and OPENAI_API_KEY must be set.")

# Initialize Firecrawl client and set OpenAI API key
firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
openai.api_key = OPENAI_API_KEY

# Pydantic model for the summarization request
class SummarizeRequest(BaseModel):
    url: str = Field(..., description="URL to scrape and summarize")
    max_scrape_length: int = Field(2000, description="Maximum characters to scrape")
    summary_max_words: int = Field(150, description="Max words for the summary")

# Endpoint for summarization
@app.post("/summarize")
def summarize(request: SummarizeRequest):
    # Step 1: Scrape the content from the URL using Firecrawl.
    try:
        response = firecrawl_app.scrape_url(url=request.url, params={"formats": ["text"]})
        if "text" not in response:
            raise HTTPException(status_code=500, detail="No text found in scraped content")
        # Limit the scraped text length.
        text = response["text"][:request.max_scrape_length]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping URL: {e}")
    
    # Step 2: Summarize the scraped text using OpenAI.
    # Construct a prompt asking to summarize in at most N words.
    prompt = f"Please summarize the following text in at most {request.summary_max_words} words:\n\n{text}"
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        summary = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing content: {e}")
    
    # Return the summary as JSON.
    return {"summary": summary}

# Optional root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Summarization Agent!"}
