import pandas as pd
import csv
import time
import random
import numpy as np


# def createBalancedData():
#     everything = db
#     alldata = db[['id', 'num_comments', 'locked']]
#     dbL  = pd.DataFrame(columns=['id', 'num_comments', 'locked'])
#     dbnL = pd.DataFrame(columns=['id', 'num_comments', 'locked'])
#     q=0
#     v=0
#     comLockedCount = 0
#     comNotLockedCount = 0
#     for index, row in alldata.iterrows():
#         if row[2]:
#             dbL.loc[q] = row
#             q+=1
#             comLockedCount += row[1]
#         elif row[2] == False:
#             dbnL.loc[v] = row
#             comNotLockedCount += row[1]
#             v+=1
#
#     sortedListLocked = dbL.sort_values(by='num_comments', ascending=False)
#     sortedListNotLocked = dbnL.sort_values(by='num_comments', ascending=False)
#     count_rowL = sortedListLocked.shape[0]
#     count_rownL = sortedListNotLocked.shape[0]
#
#     distributeL = round(comLockedCount/3)
#     distributenL = round(comNotLockedCount/3)
#
#     part1 = distributeL*0.2
#     partnL = distributenL*0.2
#
#     idnumbers1 = []
#     idnumbers2 = []
#     idnumbers3 = []
#     idnumbers4 = []
#     idnumbers5 = []
#
#
#     i=0
#     x=1
#     for index, row in sortedListLocked.iterrows():
#         if (i<part1 and x==1):
#             idnumbers1.append(row[0])
#             i += row[1]
#         elif (x == 1):
#             x += 1
#             i = 0
#
#         if (i < part1 and x == 2):
#             i += row[1]
#             idnumbers2.append(row[0])
#         elif (x == 2):
#             x += 1
#             i = 0
#
#         if (i < part1 and x == 3):
#             i += row[1]
#             idnumbers3.append(row[0])
#
#         elif (x == 3):
#             x += 1
#             i = 0
#
#         if (i < part1 and x ==4):
#             i += row[1]
#             idnumbers4.append(row[0])
#         elif (x == 4):
#             x += 1
#             i = 0
#
#         if (x ==5):
#             i += row[1]
#             idnumbers5.append(row[0])
#     i = 0
#     x = 1
#     idnumbers1NL = []
#     idnumbers2NL = []
#     idnumbers3NL = []
#     idnumbers4NL = []
#     idnumbers5NL = []
#     for index, row in sortedListNotLocked.iterrows():
#         if (i<partnL and x==1):
#             idnumbers1NL.append(row[0])
#             i += row[1]
#         elif (x == 1):
#             x += 1
#             i = 0
#
#         if (i < partnL and x == 2):
#             i += row[1]
#             idnumbers2NL.append(row[0])
#         elif (x == 2):
#             x += 1
#             i = 0
#
#         if (i < partnL and x == 3):
#             i += row[1]
#             idnumbers3NL.append(row[0])
#
#         elif (x == 3):
#             x += 1
#             i = 0
#
#         if (i < partnL and x ==4):
#             i += row[1]
#             idnumbers4NL.append(row[0])
#         elif (x == 4):
#             x += 1
#             i = 0
#
#         if (x ==5):
#             i += row[1]
#             idnumbers5NL.append(row[0])
#
#     # print(idnumbers1NL)
#     # print(idnumbers1)
#     # print(idnumbers3NL)
#     # print(idnumbers4NL)
#     # print(idnumbers5NL)
#     scaling = 0.85
#     ratio1 = len(idnumbers1NL) / len(idnumbers1) * scaling
#     ratio2 = len(idnumbers2NL) / len(idnumbers2) * scaling
#     ratio3 = len(idnumbers3NL) / len(idnumbers3) * scaling
#     ratio4 = len(idnumbers4NL) / len(idnumbers4) * scaling
#     ratio5 = len(idnumbers5NL) / len(idnumbers5) * scaling
#     print(everything.shape)
#     w = 0
#
#     for id in idnumbers1:
#
#         amnt_dupl = round((random.uniform(0, 1) - 0.5) + ratio1)
#         print(amnt_dupl)
#         for index, row in everything.iterrows():
#             if row[0] == id:
#                 for i in range(amnt_dupl):
#                     everything.loc[everything.size + 1] = row
#
#     print(everything.shape, w)
#
#     for id in idnumbers2:
#         amnt_dupl = round((random.uniform(0, 1) - 0.5) + ratio2)
#         for index, row in everything.iterrows():
#             if row[0] == id:
#                 for i in range(amnt_dupl):
#                     everything.loc[everything.size + 1] = row
#
#     print(everything.shape)
#
#     for id in idnumbers3:
#         amnt_dupl = round((random.uniform(0, 1) - 0.5) + ratio3)
#
#         for index, row in everything.iterrows():
#             if row[0] == id:
#                 for i in range(amnt_dupl):
#                     everything.loc[everything.size + 1] = row
#
#     print(everything.shape)
#
#     for id in idnumbers4:
#         amnt_dupl = round((random.uniform(0, 1) - 0.5) + ratio4)
#
#         for index, row in everything.iterrows():
#             if row[0] == id:
#                 for i in range(amnt_dupl):
#                     everything.loc[everything.size + 1] = row
#     print(everything.shape)
#
#     for id in idnumbers5:
#         amnt_dupl = round((random.uniform(0, 1) - 0.5) + ratio5)
#
#         for index, row in everything.iterrows():
#             if row[0] == id:
#                 for i in range(amnt_dupl):
#                     everything.loc[everything.size + 1] = row
#
#     #'call database and 5x idnumbers1 verdubbelen, 3x idnumbers2 en 2x idnumbers3'
#
#     print(everything.shape)
#     return(everything)

if __name__ == '__main__':
    # db = pd.read_csv("submissiondatabase1558829476.6550424.csv")
    # newbd = createBalancedData()
    # newbd.to_csv('balanced_submissiondatabase1558829476.6550424.csv', sep='\t', encoding='utf-8')
    db = pd.read_csv("balanced_submissiondatabase1558829476.6550424 (2).csv")

    lockedlist = db['locked'].to_list()
    locked_count = 0
    nonlocked_count = 0
    totalcount = 0
    for status in lockedlist:
        if status == True:
            locked_count += 1
        if status == False:
            nonlocked_count += 1
        totalcount += 1
    print('locked_count')
    print(locked_count)
    print('nonlocked_count')
    print(nonlocked_count)
    print('totalcount')
    print(totalcount)



