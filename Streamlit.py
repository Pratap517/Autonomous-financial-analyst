from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import openai
from dotenv import find_dotenv, load_dotenv
import requests
import json
import streamlit as st


# load_dotenv(find_dotenv())
# SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
# SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = "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"
SERPAPI_API_KEY = "50630e438aacdabfba9b63ad7b6d3bf6d86a0331"


def search_financials(company_name):
    url = "https://google.serper.dev/search"
    query = f"{company_name} latest financial balance sheet"
    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": SERPAPI_API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()  # Raises stored HTTPError, if one occurred.
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as err:
        st.error(f"An error occurred: {err}")
        return {}

    response_data = response.json()
    return response_data


def summarise_financial_statements(response_data, company, balance_sheet_last_year):
    response_str = json.dumps(response_data)

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7,openai_api_key=OPENAI_API_KEY)
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


# In the main function
def main():
    load_dotenv(find_dotenv())
    balance_sheet_last_year = """The SNL Financial data shows average assets of 6,762,758 for 2018 FY, 
    7,350,233 for 2019 FY, 17,253,669 for 2020 FY, 27,581,904 for 2021 FY, 
    and 226,165,612 for 2022 FY. Average debt values are 2,289,340 for 2018 FY, 
    3,130,588 for 2019 FY, 10,425,649 for 2020 FY, 14,353,017 for 2021 FY, and 15,865,766 for 2022 
    FY; average equity values are 2,849,345 for 2018 FY, 2,642,662 for 2019 FY, 4,051,418 for 2020 FY, 8,009,609 for 2021 FY, and 8,925,921 for 2022 FY."""

    st.set_page_config(page_title="Autonomous financial analyst", page_icon=":dollar:")
    st.header("Autonomous financial analyst :dollar:")
    openaiapi = st.text_input("OpenAI API Key",type='password')
    company = st.text_input("Company to analyse")

    if not openaiapi:
        st.warning("Please provide the OpenAI API key.")
        return

    if not company:
        st.warning("Please provide the company to analyse.")
        return

    openai.api_key = openaiapi

    st.write("Generating financial summary for: ", company)

    financial_data = search_financials(company)
    if financial_data:
        summary = summarise_financial_statements(
            financial_data, company, balance_sheet_last_year
        )

        with st.expander("Financial Data"):
            st.info(financial_data)
        with st.expander("Summary"):
            st.info(summary)


if __name__ == "__main__":
    main()
