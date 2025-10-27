# github_server.py
import os
import requests
from mcp.server.fastmcp import FastMCP

# Create an MCP server named "GitHub"
mcp = FastMCP("GitHub")

# 🪪 Replace with your personal access token (fine-grained GitHub token recommended)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = "https://api.github.com"


def github_headers():
    """Return headers for authenticated GitHub requests"""
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


# --- Define MCP Tools ---

@mcp.tool()
def list_repos(username: str):
    """List public repositories for a given GitHub username"""
    url = f"{BASE_URL}/users/{username}/repos"
    response = requests.get(url, headers=github_headers())
    if response.status_code != 200:
        return f"Error: {response.status_code}, {response.text}"
    repos = [repo["name"] for repo in response.json()]
    return repos


@mcp.tool()
def get_repo_info(owner: str, repo: str):
    """Get details of a specific repository"""
    url = f"{BASE_URL}/repos/{owner}/{repo}"
    response = requests.get(url, headers=github_headers())
    if response.status_code != 200:
        return f"Error: {response.status_code}, {response.text}"
    data = response.json()
    return {
        "name": data["name"],
        "description": data["description"],
        "stars": data["stargazers_count"],
        "forks": data["forks_count"],
        "language": data["language"],
    }


@mcp.tool()
def list_issues(owner: str, repo: str):
    """List open issues in a given repository"""
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues"
    response = requests.get(url, headers=github_headers())
    if response.status_code != 200:
        return f"Error: {response.status_code}, {response.text}"
    issues = [
        {"title": issue["title"], "url": issue["html_url"]}
        for issue in response.json()
    ]
    return issues


@mcp.tool()
def create_issue(owner: str, repo: str, title: str, body: str = ""):
    """Create a new issue in a repository"""
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues"
    payload = {"title": title, "body": body}
    response = requests.post(url, headers=github_headers(), json=payload)
    if response.status_code != 201:
        return f"Error: {response.status_code}, {response.text}"
    data = response.json()
    return {"issue_url": data["html_url"], "title": data["title"]}


# --- Run the server ---
if __name__ == "__main__":
    # You can also use transport="streamable_http" if you want web-based access
    mcp.run(transport="stdio")
