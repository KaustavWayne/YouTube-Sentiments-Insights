import requests

url = "https://huggingface.co/spaces/dystopiareloaded7/youtube-chrome-plugin/api/predict"
data = {
    "data": [
        "This video is great!",
        "I absolutely hate this video.",
        "I really like your teaching style. Good work"
    ]
}

response = requests.post(url, json=data)
print(response.text)
