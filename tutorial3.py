import os
from dotenv import load_dotenv
from langchain.chains.sequential import SimpleSequentialChain
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

# first chain- Generate a business idea
idea_prompt=PromptTemplate(
    input_variables=["Industry"],
    template="Generate an innovation and practical business idea in the {Industry}"
)
idea_chain=LLMChain(llm=llm,prompt=idea_prompt)

# second chain- Create a business plan outline
plan_prompt=PromptTemplate(
    input_variables=["business_idea"],
    template="""""
Create a detailed business plan outline for this business idea: {business_idea}
Include:
1. Executive Summary
2. Market Analysis
3. Revenue Model
4. Key Challenges
5. Success Metrics
"""
)

plan_chain=LLMChain(llm=llm,prompt=plan_prompt)

# third chain- Create a marketing strategy
marketing_prompt=PromptTemplate(
    input_variables=["business_plan"],
    template="""
Based on this business plan: {business_plan}
Create a marketing strategy including:
- Target audience
- Key messaging
- Marketing channels
- Budget considerations
"""
)

marketing_chain=LLMChain(llm=llm,prompt=marketing_prompt)

# Combine the chains
business_chain=SimpleSequentialChain(
    chains=[idea_chain,plan_chain,marketing_chain],
    verbose=True # show intermediate steps
)

# Run the complete business development process
print("=== Complete Business Development process===")
final_results=business_chain.run("Tech")
print("\nFinal Marketing strategy")
print(final_results)