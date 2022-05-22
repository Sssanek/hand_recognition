from csv import reader

dct = {'0': 0,
       '1': 0,
       '2': 0,
       '3': 0,
       '4': 0,
       '5': 0,
       '6': 0,
       '7': 0,
       '8': 0,
       '9': 0,
       }
with open('point_history.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        dct[row[0]] += 1
print(*dct.items(), sep='\n')
