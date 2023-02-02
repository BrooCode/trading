import csv
from datetime import datetime as dt

# (str(row[-2]))[0:4]=="2022"
stck_name = 'CUMMINSIND'
with open('data-1674364923034.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    cost=0
    with open('stocks\CUMMINSIND.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in csv_reader:
            if line_count == 0: 
                writer.writerow(["Symbol" ,"Date", "wap","lot_size","Expiry_date"])
                print()
                line_count += 1
            elif row[1]==stck_name  :
                    writer.writerow([row[1],row[-2],row[-3],row[-6],row[2]])
                    line_count += 1
    print(f'Processed {line_count} lines.')
