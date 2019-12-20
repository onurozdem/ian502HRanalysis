from HRAnalysis.models import UnprocessedData
import pandas as pd

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#insert_data = pd.read_csv(BASE_DIR + "/data/HR_data.csv")
insert_data = pd.read_csv("C:\\Users\\onuro\\Desktop\\ŞehirÜni\\VeriOdaklıProgramlama\\Gitproject\\ian502HRanalysis\\HRAnalysis\\data\\HR_data.csv")
insert_data_dict = insert_data.to_dict('records')

i = 0
for row in insert_data_dict:
    row['id'] = i
    i += 1
    m = UnprocessedData(**row)
    m.save()
