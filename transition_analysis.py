from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

import numpy as np
import pandas as pd
import openai
from llama_index.llms.openai import OpenAI
from openai import AsyncOpenAI

import glob
import json
import asyncio
import re
import os
import time
import sys


PROMPT_TEMPLATE_GENERAL = ("""
You are tasked with the role of a climate scientist, assigned to analyze a company's sustainability report. Based on the following extracted parts from the sustainability report, answer the given QUESTIONS.
If you don't know the answer, just say that you don't know by answering "NA". Don't try to make up an answer.

Given are the following sources:
--------------------- [BEGIN OF SOURCES]\n
{sources}\n
--------------------- [END OF SOURCES]\n

QUESTIONS:
1. What is the company of the report?
2. What sector does the company belong to?
3. Where is the company located?

Format your answers in JSON format with the following keys: COMPANY_NAME and COMPANY_SECTOR COMPANY_LOCATION.
Your FINAL_ANSWER in JSON (ensure there's no format error):
""")

PROMPT_TEMPLATE_YEAR = ("""
You are tasked with the role of a climate scientist, assigned to analyze a company's sustainability report. Based on the following extracted parts from the sustainability report, answer the given QUESTION.
If you don't know the answer, just say that you don't know by answering "NA". Don't try to make up an answer.

Given are the following sources:
--------------------- [BEGIN OF SOURCES]\n
{sources}\n
--------------------- [END OF SOURCES]\n

QUESTION:
In which year was the report published?

Format your answers in JSON format with the following key: YEAR
Your FINAL_ANSWER in JSON (ensure there's no format error):
""")


PROMPT_TEMPLATE_QA = ("""
You are a senior sustainabiliy analyst with expertise in climate science evaluating a company's climate-related transition plan and strategy.

This is basic information to the company:
{basic_info}

You are presented with the following sources from the company's report:
--------------------- [BEGIN OF SOURCES]\n
{sources}\n
--------------------- [END OF SOURCES]\n

Given the sources information and no prior knowledge, your main task is to respond to the posed question encapsulated in "||".
Question: ||{question}||

Please consider the following additional explanation to the question encapsulated in "+++++" as crucial for answering the question:
+++++ [BEGIN OF EXPLANATION]
{explanation}
+++++ [END OF EXPLANATION]

Please enforce to the following guidelines in your answer:
1. Your response must be precise, thorough, and grounded on specific extracts from the report to verify its authenticity.
2. If you are unsure, simply acknowledge the lack of knowledge, rather than fabricating an answer.
3. Keep your ANSWER within {answer_length} words.
4. Be skeptical to the information disclosed in the report as there might be greenwashing (exaggerating the firm's environmental responsibility). Always answer in a critical tone.
5. Cheap talks are statements that are costless to make and may not necessarily reflect the true intentions or future actions of the company. Be critical for all cheap talks you discovered in the report.
6. Always acknowledge that the information provided is representing the company's view based on its report.
7. Scrutinize whether the report is grounded in quantifiable, concrete data or vague, unverifiable statements, and communicate your findings.
8. Start your answer with a "[[YES]]"" or ""[[NO]]"" depending on whether you would answer the question with a yes or no. Always complement your judgement on yes or no with a short explanation that summarizes the sources in an informative way, i.e. provide details.

Format your answer in JSON format with the two keys: ANSWER (this should contain your answer string without sources), and SOURCES (this should be a list of the SOURCE numbers that were referenced in your answer).
Your FINAL_ANSWER in JSON (ensure there's no format error):
""")


# function that takes the report and creates the retriever (with indexes etc.)
def createRetriever(REPORT, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K):
    # load in document
    documents = SimpleDirectoryReader(input_files=[REPORT]).load_data()
    parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)  # tries to keep sentences together
    nodes = parser.get_nodes_from_documents(documents)

    # build indexes
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model
    )

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
    )
    return retriever


def basicInformation(retriever, PROMPT_TEMPLATE_GENERAL, MODEL):
    # Query content
    retrieved_nodes = retriever.retrieve(
        "What is the name of the company, the sector it operates in and location of headquarters?")
    # create the "sources" block
    sources = []
    for i in retrieved_nodes:
        page_num = i.metadata['page_label']
        # remove "\n" from the sources
        source = i.get_content().replace("\n", "")
        sources.append(f"PAGE {page_num}: {source}")
    sources_block = "\n\n\n".join(sources)

    qa_template = PromptTemplate(PROMPT_TEMPLATE_GENERAL)
    # you can create text prompt (for completion API)
    prompt = qa_template.format(sources=sources_block)

    # or easily convert to message prompts (for chat API)
    # messages = qa_template.format_messages(sources=sources_block)

    # get response
    response = OpenAI(temperature=0, model=MODEL).complete(prompt)
    # replace front or back ```json {} ```
    response_text_json = response.text.replace("```json", "").replace("```", "")
    response_text = json.loads(response_text_json)

    # create a text to it
    basic_info = f" - Company name: {response_text['COMPANY_NAME']}\n - Industry: {response_text['COMPANY_SECTOR']}\n - Headquarter Location: {response_text['COMPANY_LOCATION']}"

    #print(basic_info)
    return basic_info, response_text

def yearInformation(retriever, PROMPT_TEMPLATE_YEAR, MODEL):
    # Query content
    retrieved_nodes = retriever.retrieve(
        "In which year was the report published?")
    # create the "sources" block
    sources = []
    for i in retrieved_nodes:
        page_num = i.metadata['page_label']
        # remove "\n" from the sources
        source = i.get_content().replace("\n", "")
        sources.append(f"PAGE {page_num}: {source}")
    sources_block = "\n\n\n".join(sources)

    qa_template = PromptTemplate(PROMPT_TEMPLATE_YEAR)
    # you can create text prompt (for completion API)
    prompt = qa_template.format(sources=sources_block)

    # or easily convert to message prompts (for chat API)
    # messages = qa_template.format_messages(sources=sources_block)

    # get response
    response = OpenAI(temperature=0, model=MODEL).complete(prompt)
    # replace front or back ```json {} ```
    response_text_json = response.text.replace("```json", "").replace("```", "")
    response_text = json.loads(response_text_json)

    return response_text


def createPromptTemplate(retriever, BASIC_INFO, QUERY_STR, PROMPT_TEMPLATE_QA, EXPLANTATION, ANSWER_LENGTH):
    # Query content
    retrieved_nodes = retriever.retrieve(QUERY_STR)
    # create the "sources" block
    sources = []
    for i in retrieved_nodes:
        page_num = i.metadata['page_label']
        # remove "\n" from the sources
        source = i.get_content().replace("\n", "")
        sources.append(f"PAGE {page_num}: {source}")
    sources_block = "\n\n\n".join(sources)

    qa_template = PromptTemplate(PROMPT_TEMPLATE_QA)
    # you can create text prompt (for completion API)
    prompt = qa_template.format(basic_info=BASIC_INFO, sources=sources_block, question=QUERY_STR,
                                explanation=EXPLANTATION, answer_length=ANSWER_LENGTH)

    # or easily convert to message prompts (for chat API)
    messages = qa_template.format_messages(basic_info=BASIC_INFO, sources=sources_block, question=QUERY_STR,
                                           explanation=EXPLANTATION, answer_length=ANSWER_LENGTH)

    return prompt


def createPrompts(retriever, PROMPT_TEMPLATE_QA, BASIC_INFO, ANSWER_LENGTH, MASTERFILE):
    prompts = []
    questions = []
    for i in np.arange(0, MASTERFILE.shape[0]):
        QUERY_STR = MASTERFILE.iloc[i]["question"]
        questions.append(QUERY_STR)
        EXPLANTATION = MASTERFILE.iloc[i]["question definitions"]
        prompts.append(
            createPromptTemplate(retriever, BASIC_INFO, QUERY_STR, PROMPT_TEMPLATE_QA, EXPLANTATION, ANSWER_LENGTH))
    print("Prompts Created")
    return prompts, questions


# asynced creation of answers
async def answer_async(prompts, MODEL):
    coroutines = []
    llm = OpenAI(temperature=0, model=MODEL)
    for p in prompts:
        co = llm.acomplete(p)
        coroutines.append(co)
    # Schedule three calls *concurrently*:
    out = await asyncio.gather(*coroutines)
    # print(L)
    return out


async def createAnswersAsync(prompts, MODEL):
    # async answering
    answers = await answer_async(prompts, MODEL)
    # return
    return answers


def createAnswers(prompts, MODEL):
    # sync answering
    answers = []
    llm = OpenAI(temperature=0, model=MODEL)
    for p in prompts:
        response = llm.complete(p)
        answers.append(response)

    print("Answers Given")
    ### create HTML of it
    return answers


def outputExcel(answers, questions, prompts, REPORT, MASTERFILE, MODEL, option="", excels_path="Excels_SustReps"):
    # create the columns
    categories, ans, ans_verdicts, source_pages, source_texts = [], [], [], [], []
    subcategories = [i.split("_")[1] for i in MASTERFILE.identifier.to_list()]
    for i, a in enumerate(answers):
        try:
            # replace front or back ```json {} ```
            a = a.text.replace("```json", "").replace("```", "")
            answer_dict = json.loads(a)
        except:
            print(f"{i} with formatting error")
            try:
                answer_dict = {"ANSWER": "CAUTION: Formatting error occurred, this is the raw answer:\n" + a.text,
                               "SOURCES": "See In Answer"}
            except:
                answer_dict = {"ANSWER": "Failure in answering this question.", "SOURCES": "NA"}

        # final verdict
        verdict = re.search(r"\[\[([^]]+)\]\]", answer_dict["ANSWER"])
        if verdict:
            ans_verdicts.append(verdict.group(1))
        else:
            ans_verdicts.append("NA")

        # other values
        ans.append(answer_dict["ANSWER"])
        source_pages.append(", ".join(map(str, answer_dict["SOURCES"])))
        source_texts.append(prompts[i].split("---------------------")[1])

        if i == 0:
            category = "target"
        if i == 12:
            category = "governance"
        if i == 21:
            category = "strategy"
        if i == 45:
            category = "tracking"
        categories.append(category)

    # create DataFrame and export as excel
    df_out = pd.DataFrame(
        {"category": categories, "subcategory": subcategories, "question": questions, "decision": ans_verdicts,
         "answer": ans,
         "source_pages": source_pages, "source_texts": source_texts})
    excel_path_qa = f"./{excels_path}/" + REPORT.split("/")[-1].split(".")[0] + f"_{MODEL}" + f"{option}" + ".xlsx"
    df_out.to_excel(excel_path_qa)
    return excel_path_qa

async def main():
    print(sys.argv)
    if len(sys.argv) < 3:
        print("WRONG USAGE PATTERN!\nPlease use: 'python api_key report model [num indicators]'")
    args = sys.argv[1:]
    os.environ["OPENAI_API_KEY"] = args[0]
    openai.api_key = args[0]
    # Global paramters
    MASTERFILE = pd.read_excel("questions_masterfile_100524.xlsx")
    CHUNK_SIZE = 350
    CHUNK_OVERLAP = 50
    TOP_K = 8
    ANSWER_LENGTH = 200

    REPORT = args[1]
    MODEL = args[2]
    # if option of less is given
    try:
        less = int(args[3])
        MASTERFILE = MASTERFILE[:less].copy()
        print(f"Execution with subset of {less} indicators.")
    except:
        less = "all"
        print("Executiuon with all paramters.")

    retriever = createRetriever(REPORT, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K)
    BASIC_INFO, response_text = basicInformation(retriever, PROMPT_TEMPLATE_GENERAL, MODEL)
    year_info = yearInformation(retriever, PROMPT_TEMPLATE_YEAR, MODEL)
    response_text["YEAR"] = year_info["YEAR"]
    response_text["REPORT_NAME"] = REPORT
    print(response_text)

    prompts, questions = createPrompts(retriever, PROMPT_TEMPLATE_QA, BASIC_INFO, ANSWER_LENGTH, MASTERFILE)

    # MAKE SURE TO NOT HIT RATE LIMITS
    answers = []
    step_size = 5
    print(f"In order to avoid Rate Limit Errors, we answer {step_size} questions at a time, not everything in parallel.\nThis increases the execution time significantly but decreases the error rate. Another potential way to overcome this is to upgrade your OpenAI API key.\nFollow the tutorials on Medium to learn more.")
    for i in np.arange(0, len(prompts), step_size):
        p_loc = prompts[i:i+step_size]
        a_loc = await createAnswersAsync(p_loc, MODEL)
        answers.extend(a_loc)
        num = i+step_size
        if num > len(prompts):
            num = len(prompts)
        print(f"{num} Answers Given")
    
    excels_path = "Excel_Output"
    option = f"_topk{TOP_K}_params{less}"
    path_excel = outputExcel(answers, questions, prompts, REPORT, MASTERFILE, MODEL, option, excels_path)

# For usage on windows:
#asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
asyncio.run(main())
