from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import openai
from dotenv import find_dotenv, load_dotenv
import requests
import json
import streamlit as st
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.chat_models import ChatOpenAI

SERPAPI_API_KEY = "e19030b0bcc80f528c588cb7c7d15d441d091598"
OPENAI_API_KEY = "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"


# request to get financial statements
def search_financials(company_name):
    search = GoogleSerperAPIWrapper(serper_api_key=SERPAPI_API_KEY)

    result = search.run(f"{company_name} latest financial balance sheet")
    print("RESULTS", result)

    return result


# llm to summarise financial statements
def summarise_financial_statements(response_data, company, balance_sheet_last_year):
    response_str = json.dumps(response_data)

    llm = llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    template = """
    You are a world class financial analyst. Here is the financial data for {company}:
    
    this is current financial information : {response_str} compare it with to previous financial information : {balance_sheet_last_year}
    

    
    Please follow all of the following rules:
    1/ Make sure the content is engaging, informative with good financial data and balance sheet insights
    2/ Make sure the content is not too long, it should be no more than 10-12 lines or points
    3/ The content should address the {company} balace sheet topic very well
    4/ The content needs to be written in a way that is easy to read and understand
    5/ The content needs to give audience actionable financial advice & insights from a financial analyst perspective
    6/ The content needs to be written withouth any grammatical errors,BOLD, ITALIC
    

    SUMMARY:
    Please summarise key points from these financial statements in bullet points.
    """

    prompt_template = PromptTemplate(
        input_variables=["response_str", "company", "balance_sheet_last_year"],
        template=template,
    )

    summary_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    summary = summary_chain.predict(
        response_str=response_str,
        balance_sheet_last_year=balance_sheet_last_year,
        company=company,
    )
    print(summary)
    return summary


def main():
    load_dotenv(find_dotenv())

    # Balance sheet for last year as a string
    balance_sheet_last_year = """
    Assets: 900000
    Liabilities: 450000
    Equity: 450000
    """

    st.set_page_config(page_title="Autonomous financial analyst", page_icon=":dollar:")

    st.header("Autonomous financial analyst :dollar:")
    openaiapi = st.text_input("OpenAI API Key")
    company = st.text_input("Company to analyse")

    openai.api_key = openaiapi

    if company:
        print(company)
        st.write("Generating financial summary for: ", company)

        financial_data = search_financials(company)
        summary = summarise_financial_statements(
            financial_data, company, balance_sheet_last_year
        )

        with st.expander("Financial Data"):
            st.info(financial_data)
        with st.expander("Summary"):
            st.info(summary)


if __name__ == "__main__":
    main()
