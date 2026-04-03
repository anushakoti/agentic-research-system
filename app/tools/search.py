import requests

def search_web(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url).json()

    results = []
    for topic in response.get("RelatedTopics", []):
        if "Text" in topic:
            results.append(topic["Text"])

    return results[:5]