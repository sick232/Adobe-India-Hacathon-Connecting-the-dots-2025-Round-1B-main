-----

# **Adobe Hackathon: Round 1B - Persona-Driven Document Intelligence**

This repository contains the solution for Round 1B of the Adobe Hackathon. The project is an intelligent system that analyzes a collection of PDF documents based on a user persona and a specific task, then extracts and ranks the most relevant sections.

## **Our Approach**

This solution uses natural language processing to understand the semantic meaning of the documents and the user's query.

1.  **Text Extraction**: Content from each PDF is extracted using **PyMuPDF**.
2.  **Vector Embedding**: A pre-trained **`sentence-transformers`** model converts the user's persona, their "job-to-be-done," and each section of the documents into numerical vectors (embeddings).
3.  **Similarity Search**: We use **`scikit-learn`** to calculate the cosine similarity between the user query vector and all document section vectors.
4.  **Ranking**: The sections with the highest similarity scores are considered the most relevant and are ranked accordingly in the final output.

## **How to Build and Run**

Follow these steps to set up the project, build the Docker image, and run the solution.

### **1. Model Preparation (One-time Setup)**

This project uses a Python script to download the required pre-trained model.

1.  First, install the Python dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
2.  Next, run the `download_model.py` script. This will download the necessary model files into the `./model/` directory, making them available for the Docker build.
    ```bash
    python download_model.py
    ```

### **2. Prepare Your Input Files**

The system requires a collection of PDFs and a JSON file defining the persona and their task.

1.  Place all your PDF documents (e.g., `Breakfast Ideas.pdf`, `Dinner Ideas - Mains_1.pdf` (Collection 3)) inside the `./input` folder.

2.  Create a file named **`challenge1b_input.json`** inside the `./input` folder with the following structure:

    ```json
    {
      "persona": "Health-conscious home cook looking for quick meals",
      "job_to_be_done": "Find main course dinner recipes that are easy to prepare and use common ingredients."
    }
    ```

### **3. Build the Docker Image**

With the model and input files in place, run the following command from the project's root directory to build the Docker image:

```bash
docker build --platform linux/amd64 -t persona-analyzer:1.0 .
```

### **4. Run the Docker Container**

Use the command below to run the container. It will process the files in your `input` folder and generate a single **`challenge1b_output.json`** file in the `output` folder.

```bash
docker run --rm -v "$(pwd)/input":/app/input -v "$(pwd)/output":/app/output --network none persona-analyzer:1.0
```

## **Key Libraries Used**

  * **sentence-transformers**: For generating high-quality sentence and text embeddings.
  * **PyMuPDF**: For fast and reliable PDF text extraction.
  * **scikit-learn**: For efficient cosine similarity calculations.
  * **numpy**: For numerical operations on vector embeddings.

## **Authors**

  * Mayank Chauhan
  * Adithya Sankar Menon
  * Piyush Maurya
