import csv
from datetime import datetime as dt

# (str(row[-2]))[0:4]=="2022"
stck_name = set()
with open('data-1674364923034.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    cost=0
    for row in csv_reader:
        if line_count == 0:
            print()
            line_count += 1
        else:
            stck_name.add(row[1])
            line_count += 1
        
    print(f'Processed {line_count} lines.')
print(stck_name)
