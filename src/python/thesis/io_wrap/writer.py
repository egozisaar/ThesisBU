import csv


def write_csv_file(name, dest, title_row, data_rows):
    with open(dest + '/' + name + '.csv', 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(title_row)
        writer.writerows(data_rows)
    file.close()
