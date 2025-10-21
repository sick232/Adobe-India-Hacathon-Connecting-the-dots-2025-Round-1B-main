# run.py (Complete Corrected Version)
import os
import json
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- CONFIGURATION ---
INPUT_DIR = "./input"
OUTPUT_DIR = "./output"
MODEL_PATH = "./model"


# --- STAGE 1: DOCUMENT INGESTION AND STRUCTURING (IMPROVED) ---

def extract_structure_from_pdf(pdf_path):
    """
    Improved function to extract structure. It prioritizes the document's
    Table of Contents (TOC) and falls back to a smarter text analysis.
    """
    doc = fitz.open(pdf_path)
    outline = []
    full_text_sections = {}

    # 1. Try to get structure from the Table of Contents (most reliable)
    toc = doc.get_toc()
    if toc:
        for level, title, page_num in toc:
            title = title.strip()
            if not title:
                continue

            level_str = f"H{level}"
            section_entry = {"level": level_str, "text": title, "page": page_num}
            outline.append(section_entry)

            # Extract full text for this section
            # This logic assumes one section per page, which is a simplification
            # A more advanced version would find the end of the section
            page = doc.load_page(page_num - 1)
            full_text_sections[title] = page.get_text("text")

    # 2. If no TOC, create a single entry for the whole document as a fallback
    else:
        title = os.path.basename(pdf_path).replace('.pdf', '')
        outline.append({"level": "H1", "text": title, "page": 1})

        # --- THIS IS THE CORRECTED PART ---
        # Iterate through each page to get the full document text
        all_text = ""
        for page in doc:
            all_text += page.get_text("text") + "\n"
        full_text_sections[title] = all_text

    doc.close()
    title = os.path.basename(pdf_path)
    return {"title": title, "outline": outline}, full_text_sections


# --- STAGE 2 & 3: ANALYSIS AND SCORING ---

def run_persona_analysis():
    """
    Main analysis pipeline.
    """
    start_time = time.time()

    # Load the offline sentence-transformer model
    model = SentenceTransformer(MODEL_PATH)

    # --- DYNAMIC INPUTS (CORRECTED) ---
    # Load the persona and job from the JSON input file
    input_json_path = os.path.join(INPUT_DIR, "challenge1b_input.json")
    try:
        with open(input_json_path, 'r') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The input file was not found at {input_json_path}")
        return

    persona = input_data.get("persona", {}).get("role", "Unknown Persona")
    job_to_be_done = input_data.get("job_to_be_done", {}).get("task", "Unknown Task")

    # Generate semantic profile for the persona + job
    query_text = f"Persona: {persona}. Task: {job_to_be_done}"
    query_embedding = model.encode(query_text)

    all_sections = []

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        structure, content = extract_structure_from_pdf(pdf_path)

        if not structure["outline"]:
            continue

        # Combine section title with its full text for better embedding context
        section_texts_for_embedding = [
            item["text"] + " " + content.get(item["text"], "")
            for item in structure["outline"]
        ]

        section_embeddings = model.encode(section_texts_for_embedding)

        # Check for empty embeddings
        if len(section_embeddings) == 0:
            continue

        similarities = cosine_similarity([query_embedding], section_embeddings)[0]

        for i, item in enumerate(structure["outline"]):
            all_sections.append({
                "document": pdf_file,
                "page": item["page"],
                "section_title": item["text"],
                "score": similarities[i],
                "full_text": content.get(item["text"], "")
            })

    ranked_sections = sorted(all_sections, key=lambda x: x["score"], reverse=True)

    # Prepare the output JSON
    output_data = {
        "metadata": {
            "input_documents": [os.path.basename(doc.get('filename', '')) for doc in input_data.get('documents', [])],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "extracted_sections": [],
        "sub_section_analysis": []
    }

    # Populate output with top 5 sections
    for i, section in enumerate(ranked_sections[:5]):
        output_data["extracted_sections"].append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": i + 1,
            "page_number": section["page"],
        })

        refined_text = section["full_text"].strip().replace("\n", " ")
        if not refined_text:
            refined_text = "Content for this section is not available."

        output_data["sub_section_analysis"].append({
            "document": section["document"],
            "refined_text": refined_text,
            "page_number": section["page"]
        })

    output_filename = "challenge1b_output.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Processing complete. Output written to {output_path}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    run_persona_analysis()