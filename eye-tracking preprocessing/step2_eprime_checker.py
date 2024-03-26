import os
import openpyxl

folder_path = 'M:\\DRAMA_data\\actual\\eprime_data'  # Replace with the path to your folder

match_count = 0
dismatch_count = 0

for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):
        xlsx_filepath = os.path.join(folder_path, filename)
        wb = openpyxl.load_workbook(xlsx_filepath)
        ws = wb.active

        # Find the column with the header 'DataFile.Basename'
        datafile_basename_col = None
        for col in ws.iter_cols(values_only=True):
            if col[0] == 'DataFile.Basename':
                datafile_basename_col = col
                break

        # Check if the column was found
        if datafile_basename_col is None:
            print(f'{filename}: Column not found')
        else:
            # Access the value in the second row of the column
            datafile_basename = datafile_basename_col[1]
            if datafile_basename == filename[:-5]:
                match_count += 1
            else:
                print(f'{filename[:-5]}: DataFile.Basename does not match filename')
                dismatch_count += 1

print("match number : ", match_count)
print("dis match number : ", dismatch_count)
