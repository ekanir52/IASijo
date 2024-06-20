from pdf2image import convert_from_path
import pytesseract
import Levenshtein
from pytesseract import Output
import cv2
from PIL import Image #, ImageEnhance, ImageFilter
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import numpy as np
import os
# import PyPDF2
import re
import spacy

def write_in_file(output_text, output_file):

    try:
        with open(output_file, "w", encoding="utf-8") as text_file:
            text_file.write(output_text)
            text_file.close()

    except:
        print("Error while writing in Output file")
        return False

    return True

def fclear(checktemp=False, checkbox=False, checkimg=False):
    # Removing all the temporary boxe image files

    dir1 = './temp'
    dir2 = './temp/boxs'
    dir3 = './images'

    items1 = os.listdir(dir1)
    items2 = os.listdir(dir2)
    items3 = os.listdir(dir3)
    
    if checktemp:
        for i in [item for item in items1 if os.path.isfile(os.path.join(dir1, item))]:
            os.remove(os.path.join(dir1, i))
    if checkbox:
        for i in [item for item in items2 if os.path.isfile(os.path.join(dir2, item))]:
            os.remove(os.path.join(dir2, i))
    if checkimg:
        for i in [item for item in items3 if os.path.isfile(os.path.join(dir3, item))]:
            os.remove(os.path.join(dir3, i))

    return True

def is_image(filepath):
    try:
        im = Image.open(filepath)
    # do stuff
    except IOError:
        print("Invalid image file")
        return False

    return True

def is_pdf(filepath):
    try:
        PdfReader(filepath)
    except PdfReadError:
        print("invalid PDF file")
        return False
    else:
        pass

    return True

# convert a PDF to image with high DPI for better read
def convert_pdf_to_image(cv_pdf):
    pages = convert_from_path(cv_pdf, 400, poppler_path="./poppler-24.02.0/Library/bin")
    cv_images = []
    for k, page in enumerate(pages):
        image_path = f'./images/page_{k}.png'
        page.save(image_path, 'PNG')
        cv_images.append(image_path)

    return cv_images


# (Using OpenCV) Preprocess image using techniques like : Grayscale conversion / Thresholding(Binarization) / Denoising / Resizing
def pp_image(pp_images, alpha, beta, dil_n_iteration, dil_rect_x_y):
    
    """
    Preprocess the images of each page of the CV and return a list of boxes of the picture

    param pp_images : list of the images
    param alpha : contrast 1-3
    param beta : brightness 0-100
    param dil_n_iteration : number of iteration of the dilation
    param dil_rect_x_y : morphology of the rectangle of dilation

    return a list of boxes

    """

    dict_boxes={}
    k=0

    for pp_image in pp_images:

        image = cv2.imread(pp_image)
        origin = cv2.imread(pp_image)

        # apply grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"./temp/gray{k}.png", image)
 
        # apply contrast
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        cv2.imwrite(f"./temp/contrast{k}.png", image)

        # apply the gaussian blur
        image = cv2.GaussianBlur(image, (5,5), 2)
        cv2.imwrite(f"./temp/blur{k}.png", image)
  
        # apply the threshold
        # image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        image = cv2.adaptiveThreshold(image, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
        
        cv2.imwrite(f"./temp/thresh{k}.png", image)

        # determine the kernel for the dilation 
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, dil_rect_x_y)

        # Dilate the image to show the boxes
        image=cv2.dilate(image, kernal, iterations=dil_n_iteration)

        cv2.imwrite(f"./temp/dilate{k}.png", image)

        # Apply erosion
        # image = cv2.erode(image, kernal, iterations=5)

        # cv2.imwrite(f"./temp/eroded{k}.png", image)
        
        # Resize the image to improve OCR accuracy
        # image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        

        # find the contours of the text boxes
        boxes= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # print(boxes[0])
        boxes = boxes[0] if len(boxes)==2 else boxes[1]
        boxes = sorted(boxes, key=lambda x: cv2.boundingRect(x)[1])
        
        ###################################################################################################

        list_boxes = []
        colorBLUE=(255, 30, 26)

        for c in boxes:
            notadd=False
            x, y, w, h = cv2.boundingRect(c)
            roi = origin[y:y+h, x:x+w]
            filename_roi = f"temp/boxs/{k}_roi_{x}_{y}_{w}_{h}.png"
            data_box = (cv2.boundingRect(c), filename_roi)
            # print("DEBUG :", data_box)
            if list_boxes == []:
                list_boxes.append(data_box)
                continue

            for a in list_boxes:
                xi,yi,wi,hi = a[0]
                if xi>x+w or yi>y+h or x>xi+wi or y>yi+hi:
                    continue
                else:
                    if xi>x and x+w>xi+wi and yi>y and y+h>yi+hi:
                        list_boxes.remove(a)
                    elif x>xi and xi+wi>x+w and y>yi and yi+hi>y+h:
                        notadd = True
                        break
                    elif x>xi and xi+wi<x+w and y>yi and yi+hi>y+h :
                        inter_w=w-(x+w-xi-wi)
                        inter_h=h
                        if inter_h*inter_w > 0.3*hi*wi:
                            notadd = True
                            break
                    elif xi>x and x+w>xi+wi and yi>y and y+h>yi+hi:
                        inter_w=wi-(xi+wi-x-w)
                        inter_h=hi
                        if inter_h*inter_w > 0.3*h*w:
                            notadd = True
                            break
            if notadd:
                # print("not adding the box :", data_box)
                continue
            list_boxes.append(data_box)
        for item in list_boxes:
            xi, yi, wi, hi = item[0]
            roi = origin[yi:yi+hi, xi:xi+wi]
            cv2.rectangle(origin, (xi,yi), (xi+wi, yi+hi), colorBLUE, 2)
            cv2.imwrite(item[1], roi)



        cv2.imwrite(pp_image,origin)
        list_boxes = sorted(list_boxes, key=lambda t: (t[0][1], t[0][0]))
        dict_boxes[k] = list_boxes
        k+=1
        
    return dict_boxes

def overlapping_nms(listofboxes):

    x,y,w,h = listofboxes[0][0]

    for item in listofboxes[1:]:
        xi,yi,wi,hi = item[0]

# Extract text using OCR with specific config:
def ocr_extract_text(bounding_boxe, config=r'--oem 3 --psm 6', stroutput=True):    

    # Perform OCR on the bounding boxes images

    image = cv2.imread(bounding_boxe)
    # print("DEBUG : ", bounding_boxe)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY_INV)[1]
    
    # image = cv2.adaptiveThreshold(image, 255,
    #                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                 cv2.THRESH_BINARY, 11, 2)

    cv2.imwrite(bounding_boxe, image)

    image = Image.open(bounding_boxe)

    if stroutput:
        extracted_text = pytesseract.image_to_string(image, lang='eng+fra')
    else:
        extracted_data_dict = pytesseract.image_to_data(image, lang='eng+fra', 
                                                        output_type=Output.DICT, 
                                                        config=r'--oem 3 --psm 6')

        current_block_num = -5
        current_par_num = -5
        current_line_num = -5
        lines = []
        line_text = ''

        for i in range(len(extracted_data_dict['text'])):
            if int(extracted_data_dict['conf'][i]) > 0:

                if (extracted_data_dict['block_num'][i] != current_block_num or
                        extracted_data_dict['par_num'][i] != current_par_num or
                        extracted_data_dict['line_num'][i] != current_line_num):
                    
                    if line_text:
                        lines.append(line_text)
                        line_text = ''
                    
                    current_block_num = extracted_data_dict['block_num'][i]
                    current_par_num = extracted_data_dict['par_num'][i]
                    current_line_num = extracted_data_dict['line_num'][i]

                taken = extracted_data_dict['text'][i]
                
                line_text += taken + ' '

        if line_text:
            lines.append(line_text)

        extracted_text = '\n'.join(lines)

    return extracted_text





# Extract text from pdf file using Fitz
# def fitz_extract_text(pdf_path):
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         text_content = []

#         # Extract text from each page
#         for page_num, page in enumerate(reader.pages):
#             text = page.extract_text()
#             text = clean_text(text)
#             text_content.append(f'\n{text}\n')
#         file.close()

#     return "\n".join(text_content)

# def clean_text(_):
#     _ = re.sub(r'\s+', ' ', _)
#     _ = re.sub(r'\s+([,.!?])', r'\1', _)
#     _ = re.sub(r'\n+', '\n', _)
#     return _.strip()


# Character Error Rate (CER) using Levenshtein distance
def calculate_cer(ocr_text, ground_truth):
    distance = Levenshtein.distance(ocr_text, ground_truth)
    cer = distance / max(len(ocr_text), len(ground_truth))
    return cer


# Word Error Rate (WER) using Levenshtein distance
def calculate_wer(ocr_text, ground_truth):

    ocr_words = ocr_text.split()
    ground_truth_words = ground_truth.split()
    distance = Levenshtein.distance(" ".join(ocr_words), " ".join(ground_truth_words))
    wer = distance / max(len(ocr_words), len(ground_truth_words))
    return wer

def extract_information(text):

    # nlp = spacy.load("fr_core_news_md")
    # doc = nlp(text)

    # entities = [(ent.text, ent.label_) for ent in doc.ents]

    list_langues = ["Français", "Francais", "Anglais", "Espagnol", "Allemand", "Italien", "Chinois", "Japonais", "Russe", "Arabe", "Portugais"]
    
    skills = ["Apache Tomcat", "Jenkins", "git", "MySQL", "Junit" , "spring", "JIRA", "C#", "Visual Studio", "Oracle", "Python", "Java", "C++", "C#", "CSharp", "C Sharp", "JavaScript", "TypeScript", "SQL", "HTML", "CSS", "React", "Angular", "Vue.js", "VueJS", "Node.js", "NodeJS", "Express.js", "Django", "Flask", "PHP", "Laravel", "Spring", "Hibernate", "Git", "Docker", "Kubernetes", "Linux", "Windows", "MacOS", "AWS", "Azure", "GCP", "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Redis", "GraphQL", "REST", "SOAP", "JUnit", "Selenium", "TensorFlow", "PyTorch"]
    skills = [re.escape(skill) for skill in skills]
    
    graphic_skills = ["Photoshop", "Illustrator", "After Effect", "After Effects", "Blender", "InDesign", "PremierePro", "Cinema4D", "Suite Adobe", "Adobe", "Canva", "Figma", "Audition"]
    graphic_skills = [re.escape(graphic_skills) for graphic_skills in graphic_skills]

    pattern_langue = r"\b(" + "|".join(list_langues) + r")\b"
    pattern_phone1 = r"\b\d{10}\b"
    pattern_phone2 = r"\b\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\b"
    pattern_phone3 = r"\b\d{2}\ \d{2}\ \d{2}\ \d{2}\ \d{2}\b"
    pattern_phone4 = r"\+\d{2} \d{9}"
    pattern_phone5 = r"\b\d{13}\b"

    pattern_email = r"\S+@\S+"
 
    email = re.findall(pattern_email, text)

    if re.findall(pattern_phone1, text) != []:
        phone=re.findall(pattern_phone1, text)
    elif re.findall(pattern_phone2, text) != []:
        phone=re.findall(pattern_phone2, text)
    elif re.findall(pattern_phone3, text) != []:
        phone=re.findall(pattern_phone3, text)
    elif re.findall(pattern_phone4, text) != []:
        phone=re.findall(pattern_phone4, text)
    elif re.findall(pattern_phone5, text) != []:
        phone=re.findall(pattern_phone5, text)
    else:
        phone=[]
   

    langues = re.findall(pattern_langue, text)
    
    pattern_competences = r"\b(" + "|".join(skills) + r")\b"
    pattern_graphic_skills = r"\b(" + "|".join(graphic_skills) + r")\b"
    competence_info = re.findall(pattern_competences, text)
    competence_graph = re.findall(pattern_graphic_skills, text)
    
    seen = set()
    unique_comp_info = []
    unique_comp_graph = []
    unique_langue = []

    for c in competence_info:
        if c not in seen:
            unique_comp_info.append(c)
            seen.add(c)

    for c in competence_graph:
        if c not in seen:
            unique_comp_graph.append(c)
            seen.add(c)

    for c in langues:
        if c not in seen:
            unique_langue.append(c)
            seen.add(c)
        
    return {
        "email": email,
        "phone": phone,
        "langues": unique_langue,
        "compet_info": unique_comp_info,
        "compet_graph": unique_comp_graph
    }