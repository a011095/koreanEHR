import csv

#load data
def loadData(fileName):
    with open(fileName,'r') as csvfile:
        filereader = csv.reader(csvfile,delimiter=',',quotechar='"')
        next(filereader,None)
        output = list(filereader)
    return output

#save data
def saveData(fileName,data):
    with open(fileName,"w") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(data)