import csv
import sys, os

print([1,2]-[3,4])

# DATA_DIRECTORY = os.path.join(\
#     os.path.split(os.path.realpath(__file__))[0], "../data")

# IMG_PATH = 1
# PLATE_NUM = 2
# MODEL = 3
# COLOR = 4
# ULX = 6
# ULY = 7
# URX = 8
# URY = 9
# DRX = 10
# DRY = 11
# DLX = 12
# DLY = 13

# csv_reader = csv.reader(open(os.path.join(DATA_DIRECTORY,"label/annotation.csv"), encoding="utf-8"))

# first = True
# for row in csv_reader:
#     if first:
#         first = False
#         continue
#     assert row[ULX]<row[URX], print(row)
#     assert row[ULY]<row[DLY], print(row)
#     assert row[DLX]<row[DRX], print(row)
#     assert row[URY]<row[DRY], print(row)
#     # print(row[COLOR])
#     # exit(0)