import os
import openpyxl

folder_path = 'M:\\DRAMA_data\\actual\\eprime_data'  # Replace with the path to your folder

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith('.edat3'):
            xlsx_filename = filename.replace('.edat3', '.xlsx')
            xlsx_filepath = os.path.join(root, xlsx_filename)
            if not os.path.exists(xlsx_filepath):
                print("creating ", xlsx_filename)
                wb = openpyxl.Workbook()
                wb.save(xlsx_filepath)
