from langchain_ollama import ChatOllama

print(ChatOllama(model="llama3.2").invoke("Say hi in 5 words").content)