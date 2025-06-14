# Data Extract Transform Load (ETL) pipeline for medical protocols

Acquired cardiology protocols from [European Society of Cardiology](https://www.escardio.org/Guidelines) and saved on [this](https://drive.google.com/drive/folders/1rgaemZ4Jetyz98ivTw8fpLIndgZ2jczn?usp=sharing) Google Drive folder.

Those pdfs have later been converted using [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) and PyMuPDF into Markdown format and saved in the **/mddocs** folder. I also modified the original code to better extract and save the images, since the method indicated in the PyMuPDF4LLM documentation didn't work properly. This way, bigger and less images are extracted, better fitting this problem. Moreover, a post-processing procedure is done, eliminating repetitive useless patterns like "[.....]".

To run the `converter.py`:
```[languages=bash]
# from root directory
cd src

# Usage: python converter.py [pdf_directory] [markdown_directory] [method]
python3 converter.py ../pdfdocs ../mddocs page_breaks
```

