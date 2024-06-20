from library_functions import fclear, is_image, is_pdf, convert_pdf_to_image, pp_image, ocr_extract_text, calculate_cer, calculate_wer, extract_information, write_in_file
import pytesseract

# It is necessary to redirect the cmd to the exec to it's current folder path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
import sys

def __main__():
    
    if len(sys.argv) != 2 :
        print("Usage: .\doctr_run.py path_to_file")
        sys.exit(1)

    # Smart Data for better OCR results
    alpha = 1.2
    beta = 10
    dil_rect = (3,3)
    dil_iterations = 37

    file = sys.argv[1]
    print(f"Processing the resume {file}..")
    fclear(checkbox=True, checkimg=True, checktemp=True)

    print("Tesseract version used: ", pytesseract.get_tesseract_version(), "\n......\n....\n..\n.")
    print()
    
    
    if is_pdf(file):
        print(" Converting pdf => image ....")
        file = convert_pdf_to_image(file)
    elif is_image(file):
        file = [file]
        print("File is already an image")
    else:
        return False
    
    print(f" Preprocessing the Resume image {file.__str__()} ...")
    dict_boxes = pp_image(file, alpha=alpha, beta=beta, dil_rect_x_y=dil_rect, dil_n_iteration=dil_iterations)

    # print(dict_boxes)
    # Tesseract custom configuration : r'--oem 3 --psm 6' (psm = 4 best one so far)
    # custom_config = r'--oem 3 --psm 4'

    print(" Extracting the data from resume ... ")
    final_text = ""
    for k in dict_boxes:
        for i in range(len(dict_boxes[k])):
            temp_text = ocr_extract_text(dict_boxes[k][i][1], stroutput=False)
            if len(temp_text) > 3:      
                final_text += ocr_extract_text(dict_boxes[k][i][1], stroutput=False) +"\n\n"
            
    outputfile = f"./outputs/anir_model_output.txt"

    write_in_file(output_text=final_text, output_file=outputfile)
    information = extract_information(final_text)
    print("Email : ", information["email"])
    print("Phone : ", information["phone"])
    print("Langues : ", information["langues"])
    print("Compétence Informatiques : ", information["compet_info"])
    print("compétences Comportementales : ", information["compet_graph"])


    # with open(gtf, "r", encoding="utf-8") as text_file:
    #     ground_truth_text = text_file.read()
    #     text_file.close()

    # print(" Calculating the error rates ....")
    # cer = calculate_cer(final_text, ground_truth_text)
    # wer = calculate_wer(final_text, ground_truth_text)
    # print(f"CER : {cer}")
    # print(f"WER : {wer}")


    return True

__main__()
