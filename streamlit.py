#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import json
import gdown
from helper import SNOMEDHelper, extract_terms_with_llm, get_top_3_snomed_codes, extract_best_codes_with_llm, get_matches


accessory_files=[{
        "name":"term_embeddings.npy",
        "id":"1ykUGEk4YJuM5SUQtUhO64k2-EObJDjnk"
    },
    {
        "name":"faiss_index.index",
        "id":"1Yrz92yVRC_5uGxrievQVXF9pF4YnYpYX"
    }]



@st.cache_data
def download_large_file(file_details):
    FILE_ID = file_details["id"]
    FILE_NAME = file_details["name"]
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, FILE_NAME, quiet=False)
    return FILE_NAME

# Call the function to download the file
#file_path1 = download_large_file(accessory_files[0])
#st.success(f"File downloaded and saved as: {file_path1}")

file_path2 = download_large_file(accessory_files[1])
st.success(f"File downloaded and saved as: {file_path2}")


# Set up the SNOMEDHelper with your local file
#snomed_file = "sct2_Description_Full-en_INT_20250501.txt"
helper = SNOMEDHelper()

st.title("SNOMED CT Coder from Discharge Summary")

summary_text = st.text_area("Paste Discharge Summary", height=200)

if st.button("Extract SNOMED Codes"):
    with st.spinner("Extracting terms using LLM..."):
        terms = extract_terms_with_llm(summary_text)
        top_3_codes = get_top_3_snomed_codes(helper, terms["clinical_findings"] + terms["procedures"])
        best_codes_json = extract_best_codes_with_llm(summary_text, top_3_codes)
        best_codes = json.loads(best_codes_json)
        clinical_findings_procedures_df = pd.DataFrame(best_codes.items(), columns=["term", "snomed_concept_id"])
    st.subheader("Extracted Clinical Terms")
    st.write(terms)

    st.subheader("Top-3 SNOMED Candidates using FAISS")
    st.json(top_3_codes)

    st.subheader("Final SNOMED CT Codes Using FAISS Approach")
    df = pd.DataFrame(best_codes.items(), columns=["Term", "SNOMED CT Concept ID"])
    st.dataframe(df)
    st.subheader("SNOMED CT Codes using string match Approach")
    # clinical_findings_procedures_df = pd.DataFrame(best_codes.items(), columns=["Term", "SNOMED CT Concept ID"])
    st.dataframe(clinical_findings_df = pd.DataFrame(get_matches(terms["clinical_findings"],helper)))
    st.dataframe(procedures_df = pd.DataFrame(get_matches(terms["procedures"],helper)))
    

