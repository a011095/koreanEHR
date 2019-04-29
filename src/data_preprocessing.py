import numpy as np
from datetime import datetime


def extractData(data, d_list, d_lv, m_lv):
    """process data from the raw data.

    Args:
        data: input data
        d_list: disease list to be excluded in the processed data
        d_lv: (user defined) # of letters for disease code for processing data
        m_lv: (user defined) # of letters for medication code for processing data

    Returns:
        data: processed data,excluding records with certain diseases
        DiseaseMap: mapping table for disease(disease to index)
        DiseaseFreq: # of disease code occurence in the data
        DrugMap: mapping table for drug code (drug code to index)
        DrugFreq: # of drug code occurence in the data
    """
    DiseaseFreq = {}  # disease code frequency table
    DiseaseMap = {}
    DrugMap = {}
    DrugFreq = {}
    gIdxForDis = 0
    gIdxForDrug = 0
    exDisease = {'G20': 1, 'G21': 1, 'G22': 1, 'F023': 1}
    redundList = []
    IDList = []

    for idx1, row in enumerate(data):
        if idx1 > -1:  # and len(row) == 8:
            bCode = False
            diseaseCodes = {}
            drugCodes = {}
            # print(idx1,len(row))
            diseaseList = row[6].split(':') + row[7].split(':')
            drugList = row[8].split(':')
            # create disease code map
            for elem in diseaseList:
                code = elem if d_lv == 0 else elem[:d_lv]  # disease code (up to a certain level)
                if code != '' and code != 'NUL':
                    # delelte patients with certain disease code
                    for d_code in d_list:
                        if elem == d_code:
                            redundList.append(idx1)
                            break
                    # generate an index for each disease code for disease code map
                    if code not in exDisease:
                        if code not in diseaseCodes:
                            diseaseCodes[code] = 1
                        if code not in DiseaseMap:
                            DiseaseMap[code] = gIdxForDis
                            gIdxForDis += 1
            # create drug code map
            for elem in drugList:
                if len(elem) == 9:
                    # print('found code')
                    code = elem if m_lv == 0 else elem[:m_lv]  # drug code (up to a certain level)
                    # generate an index for each drug code for drug code map
                    if code not in drugCodes:
                        drugCodes[code] = 1
                    if code not in DrugMap:
                        # print('added code:{}'.format(code))
                        DrugMap[code] = gIdxForDrug
                        gIdxForDrug += 1
            # compute frequency of each drug code & each disease code
            for key in drugCodes:
                if key not in DrugFreq:
                    DrugFreq[key] = drugCodes[key]
                else:
                    DrugFreq[key] += drugCodes[key]
            for key in diseaseCodes:
                # print(key)
                if key not in DiseaseFreq:
                    DiseaseFreq[key] = diseaseCodes[key]
                else:
                    DiseaseFreq[key] += diseaseCodes[key]
        else:
            redundList.append(idx1)
    # delete patient information with certain disease
    for elem in range(len(redundList) - 1, -1, -1):
        del data[redundList[elem]]

    #    return output, DiseaseMap, DiseaseFreq, DrugMap, DrugFreq
    return data, DiseaseMap, DiseaseFreq, DrugMap, DrugFreq


def swap_columns(my_array, col1, col2):
    """Swap two columns

    Args:
        my_array: input data
        col1: the first column
        col2: the second column.

    Returns:
        None
    """
    for row in my_array:
        temp = row[col1]
        if isinstance(row[col2], int):
            row[col1] = row[col2]
        else:
            row[col1] = 0
        row[col2] = temp
    # return my_array


##convert drug code (new format -> old format)
def convertDrugCode(data, map, drugIdx, numElem=9):
    """convert old & new drug code to code for its main ingredient.

    Args:
        data: input data
        map: loaded a document (druc code.csv), which contains drug information.
        drugIdx: columns containing drug codes in the input data
        numElem: total number of columns in the input data

    Returns:
        output: processed data with main ingredient code.
    """

    drugCodeConvertTable = {}
    output = []
    # create mapping table (new disease code -> old disease code)
    for row in map:
        if len(row[4]) == 9:
            drugCodeConvertTable[row[4]] = row[3]
        if len(row[5]) == 9:
            drugCodeConvertTable[row[5]] = row[3]

    # convert new code to old code in data
    for row in data:
        if int(row[2]) > 11:  # 2 is column # for age. if age > 45
            newRow = []
            for idx, elem in enumerate(row):
                bFound = False
                for dIdx in drugIdx:
                    if idx == dIdx:
                        drugList = elem.split(':')
                        temp = ''
                        for drug in drugList:
                            if drug in drugCodeConvertTable:
                                if temp != '':
                                    temp += ':' + drugCodeConvertTable[drug]
                                else:
                                    temp = drugCodeConvertTable[drug]
                            else:
                                if temp != '':
                                    temp += ':' + drug
                                else:
                                    temp = drug
                        # print(temp)
                        newRow.append(temp)
                        bFound = True
                if not bFound:
                    newRow.append(elem)
            if len(newRow) < numElem:
                for i in range(numElem - len(newRow)):
                    newRow.append('NULL')
            output.append(newRow)
    return output


def cal_date(dat,start,end,target,bAdd):
    """compute total days from the first visit to the last visit

    Args:
        data: input data
        start: the first visit date.
        end: the last visit date.
        target: the target column that the computed date is saved
        bAdd: insert the target column (True) or replaced the existing column(False)

    Returns:
        None
    """
    date_format = "%Y%m%d"
    for row in dat:
        if row[start] == 'NULL' or row[start] == 0:
            if bAdd:
                row.insert(target, 0)
            else:
                row[target] = 0
        elif(int(row[start]) > 0):
            s_date = datetime.strptime(str(row[start]), date_format)
            e_date = datetime.strptime(str(row[end]), date_format)
            if bAdd:
                row.insert(target,(e_date - s_date).days)
            else:
                row[target] = (e_date - s_date).days


# filter out our PC patients without a death code,C61
def filtering(dat, dcode_idx,disease_code):
    C61PC_list = []

    for idx, row in enumerate(dat):
        if row[dcode_idx] == disease_code:
            C61PC_list.append(row)
    return C61PC_list


def mergePhyExams(dat1, dat2):
    output = np.zeros((len(dat1) + len(dat2), 35))

    for idx1, row in enumerate(dat1):
        idx = 0
        for idx2, elem in enumerate(row):
            if idx2 < 14:
                idx = idx2
            elif idx2 > 16 and idx2 < 22:
                idx = idx2 - 3
            elif idx2 > 25 and idx2 < 32:
                idx = idx2
            elif idx2 > 32 and idx2 < 36:
                idx = idx2 - 1
            output[idx1, idx] = 0 if elem == 'NULL' else int(elem)
        if len(row) > 24:
            for idx2 in range(22, 25):
                if row[idx2] == 1:
                    output[idx1, 24] = 1
                elif row[idx2] == 4:
                    output[idx1, 21] = 1
                elif row[idx2] == 5:
                    output[idx1, 20] = 1
                elif row[idx2] == 6:
                    output[idx1, 19] = 1
                elif row[idx2] == 7:
                    output[idx1, 22] = 1
                elif row[idx2] == 8:
                    output[idx1, 25] = 1
                elif row[idx2] == 9:
                    output[idx1, 25] = 1

    for idx1, row in enumerate(dat2):
        idx = 0
        for idx2, elem in enumerate(row):
            if idx2 < 32:
                idx = idx2
            elif idx2 > 34 and idx2 < 38:
                idx = idx2 - 3
            output[idx1 + len(dat1), idx] = 0 if elem == 'NULL' else int(elem)
    return output


def comDiseaseDrugDist(inData, cache, diseaseCriteria, drugCriteria, pred_year, d_lv, m_lv):
    """ process data based on drug & disease map.
    Args:
        inData:
        cache:
        diseaseCriteria:
        drugCriteria:
        pred_year:
        d_lv:
        m_lv:

    Returns:
        np_output: processed data with the drug codes.
    """
    diseaseMap, drugMap, diseaseFreq, drugFreq = cache

    offset = 0
    start = 6  # after person_id,sex,age, income, look-back period + the latest year for physical exam record
    np_output = np.zeros((len(inData), len(diseaseMap) + len(drugMap) + start),dtype=int)


    for idx, row in enumerate(inData):
        # person_id,sex,age,income, look-back period
        np_output[idx, 0] = int(inData[idx][0])  # person_id
        np_output[idx, 1] = int(inData[idx][1])  # sex
        np_output[idx, 2] = int(inData[idx][2])  # age
        np_output[idx, 3] = int(inData[idx][3])  # income
        # look-back period
        if inData[idx][4] == 'NULL':
            np_output[idx, 4] = 0;
        else:
            np_output[idx, 4] = int(inData[idx][4])
        # year for the last physical exam record
        if inData[idx][5] == 'NULL':
            np_output[idx, 5] = 2010 - pred_year
        else:
            np_output[idx, 5] = int(int(inData[idx][5 - offset]) / 10000) - pred_year


        diseaseList = row[6 - offset].split(':') + row[7 - offset].split(':')  # disease(diagnosis) code
        drugList = row[8 - offset].split(':')  # medication code

        # disease code
        for elem in diseaseList:
            code = elem if d_lv == 0 else elem[:m_lv]
            if code in diseaseMap and diseaseFreq[code] > diseaseCriteria:
                np_output[idx, diseaseMap[code] + start] += 1

        # drug code
        for elem in drugList:
            code = elem if m_lv == 0 else elem[:m_lv]
            if code in drugMap and drugFreq[code] > drugCriteria:
                np_output[idx, len(diseaseMap) + drugMap[code] + start] += 1
    return np_output


def findTagetCodeInEHR(data, pID_dic=None):
    """find recods with certain drug codes and extract only the records.

    If a list of pIDs is provided, only search records with the PIDs

    Args:
        data: input data
        pID_dic: a list of pID

    Returns:
        output: processed data with the drug codes.
    """
    tgtCode = ['1486', '2245', '3852', '1900']
    totalAD = 0
    totalADT = 0
    periods = []
    output = []
    for dat in data:
        pres_days = 0
        # print(dat[0])
        if pID_dic:
            if int(dat[0]) in pID_dic:
                drugList = dat[11].split(':')
                periodList = dat[12].split(':')
                # print(periodList)
                bAD = False
                totalAD += 1
                for idx, drug in enumerate(drugList):
                    for tgt in tgtCode:
                        if drug[:4] == tgt:
                            if not periodList[idx] == '':
                                pres_days += int(periodList[idx])
                            bAD = True
                if bAD:
                    periods.append(pres_days)
                    totalADT += 1
                    output.append(dat)
        else:
            drugList = dat[11].split(':')
            periodList = dat[12].split(':')
            # print(periodList)
            bAD = False
            totalAD += 1
            for idx, drug in enumerate(drugList):
                for tgt in tgtCode:
                    if drug[:4] == tgt:
                        if not periodList[idx] == '':
                            pres_days += int(periodList[idx])
                        bAD = True
            if bAD:
                periods.append(pres_days)
                totalADT += 1
                output.append(dat)

    print('AD based on code:{}, AD based on drug:{}'.format(totalAD, totalADT))
    np_period = np.array(periods)
    np_period = np_period[np.nonzero(np_period)]
    print('avg. prescription:{}'.format(np.mean(np_period)))
    print('min. prescription:{}'.format(np.min(np_period)))
    print('max. prescription:{}'.format(np.max(np_period)))
    print('std. prescription:{}'.format(np.std(np_period)))

    return output


def addPhyExamFeatures(pData, data, peData, phyExamLength,yearCol=0):
    """ integrate data with physical exam records.

    combine data with corresponding physical exam records (latest) based on each patient's last visit.
    Additionally, remove certain column (in current version: pID, last visit, year performed physical exam.)

    Args:
        pData: not used (TODO: delete!!!)
        data: input data
        peData: physical exam data
        yearCol: column # of the last visit.

    Returns:
        np_output: processed data (input + physical exam records).
    """
    maxCol = phyExamLength  # peData.shape[1]

    startIdx = 3  # to remove the first three columns/attributes
    # output = []

    # remove 2 field(p_id,PE year) + 6 fields (5 fields collected after 2008 + idx) + 7 personal disease history
    output = np.zeros((len(data), len(data[0]) - 2 + maxCol - startIdx - 6 - 7))
    count = 0

    for idx, row in enumerate(data):
        if yearCol == 0:
            year = 2010
        else:
            year = int(row[yearCol])
            # year =  100 * (int(row[yearCol]) - 100) // 10000

        # print(year)
        if year > 2001:
            # print(row[0])
            peRow_int = []
            if row[0] in peData[year]:  # search physical exam record from previous year
                peRow = peData[year][row[0]]

                for elem in peRow:
                    if elem == 'NULL':
                        peRow_int.append(0)
                    else:
                        peRow_int.append(int(elem))
                npPERow = np.array(peRow_int[startIdx:-1]).astype(int)
                npPERow = np.delete(npPERow,[2,7,8,9,12,16,17,18,19,20,21,22]) #delete fields collected after 2008
                output[count] = np.r_[np.delete(data[idx], [0, 5]), npPERow] #remove p_id & PE year
                count += 1

            elif year - 1 > 2001:  # search physical exam record from a year before previous year
                if row[0] in peData[year - 1]:
                    peRow = peData[year - 1][row[0]]

                    for elem in peRow:
                        if elem == 'NULL':
                            peRow_int.append(0)
                        else:
                            peRow_int.append(int(elem))
                    npPERow = np.array(peRow_int[startIdx:-1]).astype(int)
                    npPERow = np.delete(npPERow, [2, 7, 8, 9, 12,16,17,18,19,20,21,22]) #delete fields collected after 2008
                    output[count] = np.r_[np.delete(data[idx], [0, 5]), npPERow] #remove p_id & PE year
                    count += 1

    print(count)
    return output[:count]

