====================================================================
  SUBMISSION PACKAGE — Text & Image Latent-Space Explorer (P6)
====================================================================

Student Name   : Aryan Bhardwaj
Roll Number    : 06
Programme      : B.Tech – Artificial Intelligence & Analytics
University     : MIT ADT University, Pune
Project Guide  : Dr. Snehalata Funde
Project Code   : P6
Submission Date: November 06, 2026
GitHub         : https://github.com/aryanbhard/Text-Image-Latent-Space-Explorer

====================================================================
CONTENTS OF THIS SUBMISSION PACKAGE
====================================================================

submission_report/
├── 06_AryanBhardwaj_P6_Project_Report.md   ← MAIN REPORT (Markdown)
├── 06_AryanBhardwaj_P6_Demo_Script.md      ← 2-3 min demo narration script
├── README.txt                               ← This file
└── report_assets/                           ← Screenshot placeholders

====================================================================
REPORT FORMAT NOTE
====================================================================

The primary report is delivered in Markdown (.md) format.

To convert to PDF or DOCX:
  Option 1: Open the .md file in VS Code and use the
            "Markdown PDF" extension to export.
  Option 2: Upload to Google Docs (File > Open > Upload .md),
            then File > Download > PDF.
  Option 3: Run via Pandoc:
            pandoc 06_AryanBhardwaj_P6_Project_Report.md \
              -o 06_AryanBhardwaj_P6_Project_Report.pdf
  Option 4: Use https://md2pdf.netlify.app/ (paste markdown, export PDF)

====================================================================
MISSING ITEMS — ACTION REQUIRED
====================================================================

The following items MUST be completed before final submission:

1. SCREENSHOTS (report_assets/ directory)
   Add screenshots of the running application to the report_assets/ folder.
   Required screenshots:
   - Page 1: Text Embedding Explorer (PCA/UMAP scatter plot)
   - Page 2: Image Embedding Explorer (image gallery + 2D map)
   - Page 3: Combined Explorer (CLIP shared space plot)
   - Page 4: Cluster Analysis (K-Means clusters + LLM summaries)
   - Page 5: VAE Latent Space (2D scatter + decoder output image)
   - Page 6: Image Generation Studio (generated garment gallery)
   - Page 7: Attention Heatmap (from all-MiniLM-L6-v2)
   - Page 7: BERT Fill-Mask results table + bar chart
   - Page 7: GPT generation output (distilgpt2)
   - Page 8: Bias differential bar chart
   - Page 9: Evaluation metrics dashboard

   To capture screenshots:
   a) Run: python -m streamlit run app.py
   b) Open http://localhost:8501 in your browser
   c) Navigate to each page and take screenshots (Win + Shift + S on Windows)
   d) Save to submission_report/report_assets/

2. DEPLOYMENT URL (optional for submission)
   If you deploy to Streamlit Cloud, update the report line:
   "Live Application URL: [PENDING DEPLOYMENT]"
   to the actual URL.

3. SIGNATURE
   The Certificate page requires your physical signature on the printed copy.

====================================================================
HOW TO RUN THE APPLICATION
====================================================================

Prerequisites: Python 3.9–3.11, pip

1. Install dependencies:
   pip install -r requirements.txt

2. Run the app:
   python -m streamlit run app.py

3. The app opens at: http://localhost:8501

4. Run tests:
   python -m pytest tests/ -v
   (Expected: 11/11 PASSED)

====================================================================
VERIFIED TEST RESULTS (at time of submission preparation)
====================================================================

pytest tests/ -v:        11/11 PASSED (12.05 seconds)
evaluate_vae.py:         ELBO=275.29, BCE=269.69, KLD=5.61
test_page7_features.py:  ALL 3 FEATURES PASSED
test_bert_gpt_edge.py:   ALL BERT + GPT TESTS PASSED

====================================================================
