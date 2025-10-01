import requests

url = "http://localhost:8000/mcp/"

headers = {
    "Accept": "application/json,text/event-stream",  # Fixed content type (removed .text/event-stream)
}

body = {
    "jsonrpc": "2.0",  # Corrected JSON-RPC version (changed from 2.8)
    "method": "tools/list",
    "id": 1,
    "params": {}
}

response = requests.post(url, headers=headers, json=body)

# Properly formatted response handling
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))  # Added decode for bytes-to-string conversion