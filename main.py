
import sys

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from pypdf import PdfReader
load_dotenv()


# function to load the pdf and take all strings
def load_pdf(pdf_path: str):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        print("DEBUG: completed parsing PDF")
        return text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        sys.exit(1)


# chain 1: extract structured summary from raw text
def extractor_chain(raw_text: str, llm):
    print("DEBUG: running chain 1...")
    chain1_prompt = """You are a technical paper extractor.
     Extract the following from the research paper text below:
     - Abstract
     - Methodology
     - Key findings / conclusions

     Format as:

     ABSTRACT:
     <content>

     METHODOLOGY:
     <content>

     FINDINGS:
     <content>"""

    extract_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=chain1_prompt),
        HumanMessage(content=f"Here is the research paper text to analyze:\n\n{raw_text}")
    ])
    chain1_extractor = extract_prompt | llm | StrOutputParser()
    response = chain1_extractor.invoke({"raw_text": raw_text})
    print("chain 1 response: ", response)
    return response

# chain 2: simplify the structured summary into plain English for a non-expert audience
def simplify_chain(extracted_output: str, llm):
    print("chain2 running...")
    chain2_prompt = """You are an aritifical intelligence communicator.
     Take the structured paper summary below and rewrite it in plain English.
     Assume the reader is smart but not a domain expert.

     Output format:

     SIMPLE SUMMARY:
     (2–3 sentences)

     REAL-WORLD PROBLEM IT SOLVES:
     (1–2 sentences)

     WHY THE METHODOLOGY MATTERS:
     (1–2 sentences)"""

    simplify_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=chain2_prompt),
        HumanMessage(content=f"here is the structured paper summary: {extracted_output}")
    ])

    chain2_simplifier = simplify_prompt | llm | StrOutputParser()
    response = chain2_simplifier.invoke({"extracted_output": extracted_output})
    print("chain 2 response: ", response)
    return response

# chain 3: based on the simplified summary, provide a relevance score and recommendation for who should read the paper and whether it's worth reading
def recommend_chain(simplified_output: str, llm):
    print("chain3 running...")
    chain3_prompt = """You are a research advisor.
     Based on the simplified summary below, rate the paper for an AI engineer.

     Output format:

     RELEVANCE SCORE (1-10):
     <number>

     WHO SHOULD READ IT:
     (e.g., ML engineers, students, domain experts, product managers)

     SHOULD I READ THIS?:
     (2–3 line verdict)"""
    recommend_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=chain3_prompt),
        HumanMessage(content=f"here is the sumplified summary {simplified_output}")
    ])
    chain3_recommender = recommend_prompt | llm | StrOutputParser()
    response = chain3_recommender.invoke({"simplified_output": simplified_output})
    print("chain 3 response: ", response)
    return response


if __name__ == '__main__':
    pdf_path = "Attenion-is-all-you-need.pdf" # pdf file path
    pdf_text = load_pdf(pdf_path)
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    full_chain = (RunnablePassthrough()
                  | RunnableLambda(lambda x: extractor_chain(x["raw_text"], llm))
                  | RunnableLambda(lambda x: simplify_chain(x, llm))
                  | RunnableLambda(lambda x: recommend_chain(x, llm)))

    result = full_chain.invoke({"raw_text": pdf_text})
    print(result)