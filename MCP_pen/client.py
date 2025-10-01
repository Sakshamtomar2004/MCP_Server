import requests

URL="http://localhost:8000/mcp"
PAYLOAD={
    "jsonrpc":"2.0",
    "method":"tools/list",
    "params":{},
    "id":1
}
HEADERS={
    "Content-Type":"application/json",
    "Accept":"application/json,text/event-stream"
}

response=requests.post(URL,json=PAYLOAD,headers=HEADERS,stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))