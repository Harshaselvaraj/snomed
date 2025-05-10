#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import faiss
import re
import openai
import pandas as pd
from sklearn.preprocessing import normalize
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

class SNOMEDHelper:
    def __init__(self, desc_file_path: str):
        self.desc_df = self._load_snomed_descriptions(desc_file_path)
        self.term_embeddings = self._build_embeddings_index()
        self.index = self._build_faiss_index()

    def _load_snomed_descriptions(self, path: str) -> pd.DataFrame:
        desc_df = pd.read_csv(path, sep="\t", dtype=str)
        desc_df = desc_df[desc_df['active'] == '1']
        desc_df['type'] = desc_df['typeId'].map({
            '900000000000003001': 'FSN',
            '900000000000013009': 'synonym'
        })
        desc_df = desc_df[['conceptId', 'term', 'type']].drop_duplicates().reset_index(drop=True)
        desc_df['term'] = desc_df['term'].astype(str).str.lower()
        return desc_df

    def _build_embeddings_index(self):
        terms = self.desc_df['term'].tolist()
        embeddings = model.encode(terms, show_progress_bar=True, batch_size=48)
        return normalize(np.array(embeddings).astype("float32"), axis=1)

    def _build_faiss_index(self):
        index = faiss.IndexFlatIP(self.term_embeddings.shape[1])
        index.add(self.term_embeddings)
        return index

    def search_snomed(self, term: str, top_k: int = 3) -> List[Dict[str, str]]:
        query_vec = model.encode([term])
        query_vec = normalize(query_vec, axis=1).astype("float32")
        _, indices = self.index.search(query_vec, top_k)

        return [
            {
                "conceptId": self.desc_df.iloc[i]['conceptId'],
                "term": self.desc_df.iloc[i]['term'],
                "type": self.desc_df.iloc[i]['type']
            }
            for i in indices[0]
        ]

    def find_close_match(self, term: str, max_matches: int = 1) -> List[Tuple[str, str, str]]:
        matches = get_close_matches(term.lower(), self.desc_df['term'], n=max_matches, cutoff=0.9)
        results = []
        for match in matches:
            row = self.desc_df[self.desc_df['term'] == match]
            if not row.empty:
                results.append((
                    row['conceptId'].values[0],
                    row['term'].values[0],
                    row['type'].values[0]
                ))
        return results

def find_snomed_matches_difflib(self,term, max_matches=1):
    # Ensure that term is a string and convert to lowercase
    term = str(term).lower()

    # Convert desc_df['term'] to string (if it's not already) and lowercase
    self.desc_df['term'] = self.desc_df['term'].astype(str).str.lower()

    # Get close matches
    matches = get_close_matches(term, self.desc_df['term'], n=max_matches, cutoff=0.9)
    
    results = []
    for match in matches:
        # Check if the type is 'synonym' or 'FSN'
        matched_row = self.desc_df[self.desc_df['term'] == match]
        
        if len(matched_row) > 0:
            concept_id = matched_row['conceptId'].values[0] if 'conceptId' in matched_row else None
            fsn = matched_row['FSN'].values[0] if 'FSN' in matched_row else None
            synonym = matched_row['term'].values[0] if 'term' in matched_row else None
            match_type = matched_row['type'].values[0] if 'type' in matched_row else 'unknown'

            # If it's a synonym, check if FSN is available
            if match_type == 'synonym' and not fsn:
                fsn = synonym  # Use the synonym as FSN if FSN is not available

            results.append((concept_id, fsn, match_type))
        
    return results


def extract_terms_with_llm(summary_text: str) -> Dict[str, List[str]]:
    prompt = f"""
You are an expert clinical language model. Analyze the following discharge summary and extract:
1. Clinical Findings — diagnoses, symptoms, observations, test results
2. Procedures — past/current procedures

Instructions:
- Expand all abbreviations (e.g., "GI" → "gastrointestinal")
- Remove duplicates and ensure terms are SNOMED-compatible

Discharge Summary:
\"\"\"
{summary_text}
\"\"\"

Format your response exactly as:
clinical_findings = [term1, term2, ...]
procedures = [term1, term2, ...]
"""
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=500,
        temperature=0.0
    )
    output = response.choices[0].message.content

    clinical_match = re.search(r'clinical_findings\s*=\s*\[(.*?)\]', output, re.DOTALL)
    procedures_match = re.search(r'procedures\s*=\s*\[(.*?)\]', output, re.DOTALL)

    if not clinical_match or not procedures_match:
        raise ValueError("Failed to parse the LLM output.")

    clinical_list = [item.strip() for item in clinical_match.group(1).split(',')]
    procedures_list = [item.strip() for item in procedures_match.group(1).split(',')]

    return {
        "clinical_findings": clinical_list,
        "procedures": procedures_list
    }

def get_top_3_snomed_codes(helper: SNOMEDHelper, terms: List[str]) -> List[Dict]:
    results = []
    for term in terms:
        matches = helper.search_snomed(term)
        results.extend(matches)
    return results

def get_matches(match_list):
    all_matches = []
    for term in match_list:
        matches = helper.find_snomed_matches_difflib(term)
        all_matches.extend(matches)
    return all_matches 

def extract_best_codes_with_llm(summary_text: str, top_3_codes: List[Dict]) -> Dict[str, str]:
    prompt = f"""
You are a medical coding assistant. Your job is to choose the most appropriate SNOMED CT code for each clinical term given a discharge summary.

Discharge Summary:
\"\"\"
{summary_text}
\"\"\"

Top-3 SNOMED candidates per term:
\"\"\"
{top_3_codes}
\"\"\"

Please return a JSON dictionary where each key is a clinical term and the value is the chosen SNOMED CT conceptId.
"""
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=1000,
        temperature=0.0
    )
    return response.choices[0].message.content  # You can parse to dict if needed

