import pytesseract
import cv2
from PIL import Image
import os
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# text
def text(text):
    print(text)

# image
def image(imagepath):

    image = cv2.imread(imagepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(image)
    print(text)
    
    
# pdf
def pdf(pdfpath):
    from pdf2image import convert_from_path

    filePath = 'pdf_select.pdf'
    doc = convert_from_path(filePath)
    path, fileName = os.path.split(filePath)
    fileBaseName, fileExtension = os.path.splitext(fileName)

    for page_number, page_data in enumerate(doc):
        txt = pytesseract.image_to_string(Image.fromarray(page_data)).encode("utf-8")
        print("Page # {} - {}".format(str(page_number),txt))


# video and audio

import whisper
def media(audiopath):
    model = whisper.load_model("base.en")
    result = model.transcribe(audiopath)
    print(result["text"])
   
media('inputs/audio-2.mp3') 
# website

import urllib.request 
from bs4 import BeautifulSoup
url = "https://en.wikipedia.org/wiki/Indian_Rebellion_of_1857" 
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html)

for script in soup(["script", "style","a","<div id=\"bottom\" >"]):
    script.extract()    

text = soup.findAll(text=True)
for p in text:
    print(p)
    