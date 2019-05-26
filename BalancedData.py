def createBalancedData(db):

if __name__ == '__main__':
    db = pd.read_csv("submissiondatabase1558829476.6550424.csv")
    newbd = createBalancedData(db)
    lockedlist = newbd['locked'].to_list()
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
