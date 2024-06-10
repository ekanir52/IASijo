from pdf2image import convert_from_path
import pytesseract
import Levenshtein
# from pytesseract import TesseractError
import cv2
from PIL import Image #, ImageEnhance, ImageFilter
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import numpy as np
import os


def fclear():
    # Removing all the temporary boxe image files

    dir1 = './temp'
    dir2 = './temp/boxs'
    dir3 = './images'

    items1 = os.listdir(dir1)
    items2 = os.listdir(dir2)
    items3 = os.listdir(dir3)

    for i in [item for item in items1 if os.path.isfile(os.path.join(dir1, item))]:
        os.remove(os.path.join(dir1, i))

    for i in [item for item in items2 if os.path.isfile(os.path.join(dir2, item))]:
        os.remove(os.path.join(dir2, i))

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
        image = cv2.GaussianBlur(image, (7,7), 0)
        cv2.imwrite(f"./temp/blur{k}.png", image)

        # apply the threshold
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        cv2.imwrite(f"./temp/thresh{k}.png", image)

        # determine the kernel for the dilation 
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, dil_rect_x_y)

        # Dilate the image to show the boxes
        image=cv2.dilate(image, kernal, iterations=dil_n_iteration)

        cv2.imwrite(f"./temp/dilate{k}.png", image)

        # Resize the image to improve OCR accuracy
        # image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        
        # find the contours of the text boxes
        cnts= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts)==2 else cnts[1]
        cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])


        list_boxes = []

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)

            roi = origin[y:y+h, x:x+w]
            filename_roi = f"temp/boxs/{k}_roi_{y}.png"

            cv2.imwrite(filename_roi, roi)

            cv2.rectangle(origin, (x,y), (x+w, y+h), (255, 30, 26), 2)

            list_boxes.append(cv2.boundingRect(c))
        
        cv2.imwrite(pp_image,origin)

        list_boxes = sorted(list_boxes, key=lambda x: x[1])
        dict_boxes[k] = list_boxes
        k+=1
    

    return dict_boxes


# Extract text using OCR with specific config:
def ocr_extract_text(pp_images, config=r'--oem 3 --psm 6'):

    cv_text=""

    for pp_img in pp_images:

        # It is necessary to redirect the cmd to the exec to it's current folder path
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        
        # Open the preprocessed image file using PIL ( except when using pp2 )
        try:
            image = Image.open(pp_img)
            print(" Using PP1... ")

        except:
            print(" Using PP2... ")
            text = pytesseract.image_to_string(pp_img, lang='fra')
            cv_text += text + "\n"
            continue


        # Perform OCR on the preprocessed image
        text = pytesseract.image_to_string(image, lang='eng+fra')

        cv_text += text + "\n"

    with open("./outputs/outputOCR.txt", "w", encoding="utf-8") as text_file:
        text_file.write(cv_text)
        text_file.close()

    print(" File saved to: ./outputs/outputOCR.txt ")
    # print(cv_text)
    return cv_text



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
