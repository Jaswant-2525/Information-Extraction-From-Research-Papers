**ABSTRACTION**

_The input to the system is a PDF research paper. The file is first converted into image format page-by-page using pdf2image. OCR is then applied using pytesseract to extract raw textual content from each page. Once text and bounding box coordinates are obtained, the LayoutLM model processes the tokens along with their positional layout to generate label predictions.

After processing all pages, the predicted tokens are combined into meaningful text segments. Further NLP processing is carried out using SpaCy to perform named entity recognition, helping identify people, organizations, dates, places, and scientific terms. Finally, the extracted information is formatted into a structured JSON output, allowing it to be reused in applications such as summarization, document classification, or research indexing systems._
