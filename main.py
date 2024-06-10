from library_functions import fclear, is_image, is_pdf, convert_pdf_to_image, pp_image, ocr_extract_text, calculate_cer, calculate_wer
# import os, shutil


def __main__():
    
    fclear()

    alpha = 1.2
    beta = 15
    dil_rect = (7,6)
    dil_iterations = 25

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
    image = pp_image(file, alpha=alpha, beta=beta, dil_rect_x_y=dil_rect, dil_n_iteration=dil_iterations)

    # print(image)
    # print(" Extracting the data from resume ... ")
    
    # # Tesseract custom configuration : r'--oem 3 --psm 6' (psm = 4 best one so far)
    # custom_config = r'--oem 3 --psm 4'
    # ocr_text = ocr_extract_text(image, custom_config)

    # print(" Calculating the error rates ....")
    # cer = calculate_cer(ocr_text, ground_truth_text)
    # wer = calculate_wer(ocr_text, ground_truth_text)


    # print(f"CER : {cer}")
    # print(f"WER : {wer}")

    return True
    


__main__()
