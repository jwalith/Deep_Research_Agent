import requests
import os

BASE_URL = "http://localhost:8000"

def test_upload():
    # Create a dummy file
    with open("test.txt", "w") as f:
        f.write("LangChain is a framework for developing applications powered by language models.")
    
    files = {'file': open('test.txt', 'rb')}
    response = requests.post(f"{BASE_URL}/upload", files=files)
    print("Upload Response:", response.json())
    
    os.remove("test.txt")

def test_chat():
    payload = {"message": "What is LangChain?", "history": []}
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print("Chat Response:", response.json())

if __name__ == "__main__":
    # Note: Server must be running for this to work.
    # This script is just for manual verification or CI.
    print("To run this test, ensure the backend is running on port 8000.")
    test_upload()
    test_chat()
