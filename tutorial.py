import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load envirment variable
load_dotenv()

# Initialize gemini
llm=ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    temperature=0.7,
    google_api_key=os.getenv('GEMINI_API_KEY')
)

# Testing the connection
print("Testing the gemini connection")
response=llm.invoke("Hello! can you introduce yourself in a funny way?")
print("AI Response:",response.content)
print("\n"+ "="*50 + "\n")