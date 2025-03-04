"""
Simple API Client for the Firecrawl Agent API

Usage:
  # To test the root endpoint:
  python api_client.py --base-url "https://fastapi-git-main-divinealways-projects.vercel.app/" --command root

  # To run the agent endpoint:
  python api_client.py --base-url "https://fastapi-git-main-divinealways-projects.vercel.app/" --command run-agent \
       --url "https://blofin.com" --prompt "Scrap and format each sentence as a separate line in a markdown list" \
       --output-file-path "scraped_content.md" --compute-limit 1
"""

import argparse
import json
import requests

def get_root(base_url: str):
    url = base_url.rstrip("/")
    try:
        response = requests.get(url)
        response.raise_for_status()
        print("GET / response:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Error calling GET /:", e)

def run_agent(base_url: str, scrape_url: str, prompt: str, output_file_path: str, compute_limit: int):
    api_url = base_url.rstrip("/") + "/run-agent"
    payload = {
        "url": scrape_url,
        "prompt": prompt,
        "output_file_path": output_file_path,
        "compute_limit": compute_limit
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        print("POST /run-agent response:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Error calling POST /run-agent:", e)
        if response is not None:
            print(response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple API Client for the Firecrawl Agent API")
    parser.add_argument("--base-url", type=str, required=True,
                        help="Base URL of the API (e.g. https://fastapi-xxxx.vercel.app)")
    parser.add_argument("--command", type=str, choices=["root", "run-agent"], required=True,
                        help="Command to execute: 'root' to test GET / or 'run-agent' to run the agent")
    parser.add_argument("--url", type=str, help="The URL to scrape (required for run-agent)")
    parser.add_argument("--prompt", type=str, help="The prompt for filtering/formatting (required for run-agent)")
    parser.add_argument("--output-file-path", type=str, default="scraped_content.md",
                        help="The key/name to store scraped content in memory (default: scraped_content.md)")
    parser.add_argument("--compute-limit", type=int, default=1,
                        help="Compute limit / maximum iterations for the agent (default: 1)")
    args = parser.parse_args()

    if args.command == "root":
        get_root(args.base_url)
    elif args.command == "run-agent":
        if not args.url or not args.prompt:
            print("Error: --url and --prompt must be provided for run-agent command")
        else:
            run_agent(args.base_url, args.url, args.prompt, args.output_file_path, args.compute_limit)
