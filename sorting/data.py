import csv

def load_access():

    X = []
    Y = []

    csv_file = open('access.csv', 'rb')
    read = csv.reader(csv_file)
    read.next()

    for home,como_funciona,contato,comprou in read:
        data = [int(home), int(como_funciona), int(contato)]

        X.append(data)
        Y.append(int(comprou))

    return X,Y

def load_search():

    X = []
    Y = []

    csv_file = open('search.csv', 'rb')
    read = csv.reader(csv_file)
    read.next()

    for home,busca,logado,comprou in read:
        data = [int(home), busca, int(logado)]

        X.append(data)
        Y.append(int(comprou))

    return X,Y
