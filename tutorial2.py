import os
from dotenv import load_dotenv
from langchain.chains.summarize.map_reduce_prompt import prompt_template
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

# Create a prompt template
prompt_template=PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms that a beginner can understand. Keep it concise"
)

# Create Chain
chain=LLMChain(
    llm=llm,
    prompt=prompt_template
)

# use the chain
result=chain.run("nlp")
print("Topic Explanation:")
print(result)