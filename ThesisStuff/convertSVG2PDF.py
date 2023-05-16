import os
import sys
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


def convert_svg_to_pdf(svg_path, pdf_path):
    drawing = svg2rlg(svg_path)
    renderPDF.drawToFile(drawing, pdf_path)


def convert_folder_svg_to_pdf(folder_path):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.svg'):
                svg_path = os.path.join(root, filename)
                pdf_path = os.path.splitext(svg_path)[0] + '.pdf'
                convert_svg_to_pdf(svg_path, pdf_path)


if __name__ == '__main__':
    dim = input('type in the dimesions the svgs are in!\n')
    folder = input('type in the folder you want to convert the svgs to pdfs!\n')
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), f'{dim}D_Saves', folder)
    convert_folder_svg_to_pdf(path)