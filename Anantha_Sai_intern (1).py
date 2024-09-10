#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install pytesseract pillow spacy opencv-python')


# In[9]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[12]:


import pytesseract
from PIL import Image
import cv2
import spacy
import matplotlib.pyplot as plt
import re

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Loaded spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess the extracted text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\n+', ' ', text)  # Replace new lines with a space
    text = re.sub(r'[^\w\s,.]', '', text)  # Remove special characters
    return text.strip()

# Function to extract text from an image using Tesseract OCR
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    text = pytesseract.image_to_string(pil_image)
    return text

# Function to perform NLP tasks on the extracted text
def analyze_text(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to apply custom rules to correct specific entity types
def custom_post_processing(text):
    # Example rules to correct specific entity types
    corrected_entities = []

    # Rules for date-like patterns
    date_patterns = re.findall(r'\d{1,2} \w{3} \d{2,4}', text)
    for date in date_patterns:
        corrected_entities.append((date, 'DATE'))

    # Rules for cardinals and other specific patterns
    cardinals_patterns = re.findall(r'\b\d{1,4}\b', text)
    for cardinal in cardinals_patterns:
        corrected_entities.append((cardinal, 'CARDINAL'))
    
    return corrected_entities

# Path to the image
image_path = r'C:\Users\ananthsairavulapati\Downloads\sample for intern 2.png'

# Extracting text from the image
text_from_image = extract_text_from_image(image_path)

# Preprocessing the extracted text
clean_text = preprocess_text(text_from_image)

# here Performing NLP analysis on the cleaned text
entities = analyze_text(clean_text)

# Applying customized post-processing rules
custom_entities = custom_post_processing(clean_text)

# Displaying the extracted text and entities
print("Extracted Text:")
print(clean_text)
print("\nNamed Entities (spaCy):")
for entity in entities:
    print(f"{entity[0]} ({entity[1]})")

print("\nNamed Entities (Custom Rules):")
for entity in custom_entities:
    print(f"{entity[0]} ({entity[1]})")

# to Display the image
image = cv2.imread(image_path)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Input Image')
plt.axis('off')
plt.show()


# In[ ]:




