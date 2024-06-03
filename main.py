from library_functions import is_image, is_pdf, convert_pdf_to_image, pp_image, pp_image2, ocr_extract_text, calculate_cer, calculate_wer
import os, shutil


def __main__():

    file = "./pdf_cv/cv_1.pdf"

    with open("./gts/ground_truth.txt", "r", encoding="utf-8") as text_file:
        ground_truth_text = text_file.read()
        text_file.close()
    
    if is_pdf(file):
        print(" Converting pdf => image ....")
        file = convert_pdf_to_image(file)
    elif is_image(file):
        file = [file]
        print("File is already an image")
    else:
        return False

    print(" Preprocessing the Resume image ...", file.__str__())
    # image = pp_image(file)
    image = pp_image2(file)
    print(" Extracting the data from resume ... ")
    ocr_text = ocr_extract_text(image)

    print(" Calculating the error rates ....")
    cer = calculate_cer(ocr_text, ground_truth_text)
    wer = calculate_wer(ocr_text, ground_truth_text)


    print(f"CER : {cer}")
    print(f"WER : {wer}")



    return True
    


__main__()