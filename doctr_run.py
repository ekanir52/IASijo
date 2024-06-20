from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from library_functions import write_in_file
# import matplotlib.pyplot as plt
import sys

def main():

    if len(sys.argv) != 2 :
        print("Usage: .\doctr_run.py path_to_file")
        sys.exit(1)
    
    file_path = sys.argv[1]

    print("Initializing the training model:")
    
    # Getting the pre-trained model
    # best accuracy - long à la détente
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='master', pretrained=True)

    # middle - rapide
    # model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

    # basic model - rapide
    # model = ocr_predictor(pretrained=True)
    
    resume = DocumentFile.from_pdf(file_path)

    result = model(resume)
    result.show()

    final_text = result.render()

    write_in_file(output_text=final_text, output_file="./outputs/doctr_output")

    return True

main()