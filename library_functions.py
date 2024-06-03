from pdf2image import convert_from_path
import pytesseract
import Levenshtein
from pytesseract import TesseractError
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from pypdf import PdfReader
from pypdf.errors import PdfReadError


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
    pages = convert_from_path(cv_pdf, 300, poppler_path="./poppler-24.02.0/Library/bin")
    cv_images = []
    for k, page in enumerate(pages):
        image_path = f'./images/page_{k}.png'
        page.save(image_path, 'PNG')
        cv_images.append(image_path)

    return cv_images



# (Using OpenCV) Preprocess image using techniques like : Grayscale conversion / Thresholding(Binarization) / Denoising / Resizing
def pp_image(pp_images):

    list_pp_images=[]
    k=0

    for pp_image in pp_images:

        image = cv2.imread(pp_image)
        
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, 6)
        
        # Apply Gaussian blur to reduce noise and improve OCR accuracy
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Apply adaptive thresholding to binarize the image
        threshold_image = cv2.adaptiveThreshold(blurred_image, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
        
        # Resize the image to improve OCR accuracy
        resized_image = cv2.resize(threshold_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        
        # Save the preprocessed image temporarily
        preprocessed_image_path = f'./pp_images/preprocessed_image{k}.png'

        cv2.imwrite(preprocessed_image_path, resized_image)

        list_pp_images.append(preprocessed_image_path)
        k=+1

    
    return list_pp_images



# (using PIL) Preprocess image using techniques like : Grayscale conversion / Thresholding(Binarization) / Denoising / Resizing
def pp_image2(pp_images):
    
    list_pp_images=[]
    k=0
    
    for pp_image in pp_images:
        image = Image.open(pp_image)

        # Convert the image to grayscale
        print(" Converting image to grayscale...")
        gray_image = image.convert('L')
        
        # Appliquer un filtre de netteté
        print(" Sharpening the image...")
        sharp_image = gray_image.filter(ImageFilter.SHARPEN)
        
        # Améliorer le contraste
        print(" Enhancing the contrast...")
        enhancer = ImageEnhance.Contrast(sharp_image)
        enhanced_image = enhancer.enhance(2)

        # Appliquer un seuillage adaptatif
        print(" Applying threshhold ...")
        threshold_image = enhanced_image.point(lambda p: p > 128 and 255)
        
        threshold_image.save(fp=f'./pp_images/preprocessed_image{k}.png')
        list_pp_images.append(threshold_image)

    return list_pp_images



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
        # Tesseract custom configuration : r'--oem 3 --psm 6'
        custom_config = r'--oem 3 --psm 6'

        # Perform OCR on the preprocessed image
        text = pytesseract.image_to_string(image, lang='eng+fra', config=custom_config)

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


# Test OCR accuracy by comparing OCR output with ground truth text
# def test_ocr_accuracy(image, ground_truth_text):

#     ocr_text = ocr_extract_text(image)
    
#     cer = calculate_cer(ocr_text, ground_truth_text)
#     wer = calculate_wer(ocr_text, ground_truth_text)
    
#     return {
#         'CER': cer,
#         'WER': wer
#     }

# print(f"CER: {test_ocr_accuracy(image, ground_truth_text)['CER']:.2f}")
# print(f"WER: {test_ocr_accuracy(image, ground_truth_text)['WER']:.2f}")