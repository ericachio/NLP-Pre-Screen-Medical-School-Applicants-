import pandas as pd
import numpy as np
import datetime
import statistics 

def replace_newline(df):
    df['APP_YEAR'] = df['APP_YEAR'].replace('\n', '').astype(int)
    df['TOTAL_HOURS'] = df['TOTAL_HOURS'].replace('\n', '').astype(float)
    df['APTITUDE_SCORE'] = df['APTITUDE_SCORE'].replace('\n', '').astype(float)
    return df

def intoDateTime(string):
    date = string.split("-")
    months = ['Holder', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month = months.index(date[1])
    year = int(date[2])
    if year > 30:
        year = year + 1900
    else:
        year = year + 2000
    return (datetime.datetime(year, month, 1))
    
def months (start_date, end_date):
    num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    return num_months

def process_months(df):
    MONTHS = []
    for index, row in df.iterrows():
        MONTHS.append(months(intoDateTime(row.START_DATE), intoDateTime(row.END_DATE)))
    
    df['MONTHS'] = MONTHS 
    return df

def cal_z_score(faculty_id, score, faculty_ID):
    # on the server
#     MEAN = faculty_ID.loc[faculty_ID['FACULTY_ID'] == faculty_id, 'MEAN'].iloc[0]
#     STANDARD_DEVIATION = faculty_ID.loc[faculty_ID['FACULTY_ID'] == faculty_id, 'STANDARD_DEVIATION'].iloc[0]

    # on local comp
    MEAN = faculty_ID[faculty_ID['FACULTY_ID'] == faculty_id]['MEAN']
    STANDARD_DEVIATION = faculty_ID[faculty_ID['FACULTY_ID'] == faculty_id]['STANDARD_DEVIATION']
    z = (score - MEAN) / STANDARD_DEVIATION
    return z

def z_score(df):
    faculty_ID = df[['FACULTY_ID','APTITUDE_SCORE']]
    faculty_ID = faculty_ID.dropna()
    
    # on the server
#     faculty_ID = faculty_ID.groupby(['FACULTY_ID']).agg({'APTITUDE_SCORE':  {'Mean': np.mean, 'Standard Deviation': statistics.stdev}}).reset_index()
#     faculty_ID.columns = faculty_ID.columns.droplevel()
#     faculty_ID = faculty_ID.rename(columns={faculty_ID.columns[0]: "FACULTY_ID", faculty_ID.columns[1]: "MEAN", faculty_ID.columns[2]: "STANDARD_DEVIATION"})
#     df['Z_SCORE'] = df.apply(lambda row: cal_z_score(row['FACULTY_ID'], row['APTITUDE_SCORE'], faculty_ID), axis=1)
    
    # on local comp
    faculty_ID['APTITUDE_SCORE'] = faculty_ID['APTITUDE_SCORE'].astype(float)
    faculty_ID = faculty_ID.groupby(['FACULTY_ID']).agg({'APTITUDE_SCORE': lambda col: col.tolist()}).reset_index()
    faculty_ID['MEAN'] = [np.mean(x) for x in faculty_ID.APTITUDE_SCORE]
    faculty_ID['STANDARD_DEVIATION'] = [statistics.stdev(x) for x in faculty_ID.APTITUDE_SCORE]
    df['Z_SCORE'] = df.apply(lambda row: cal_z_score(row['FACULTY_ID'], float(row['APTITUDE_SCORE']), faculty_ID), axis=1)
    
    
    return df

def faculty_mean_diff(df):
    mean = df["APTITUDE_SCORE"].mean(axis=0) 
    
    faculty_ID = df[['FACULTY_ID','APTITUDE_SCORE']]
    faculty_ID = faculty_ID.groupby(['FACULTY_ID']).agg({'APTITUDE_SCORE':  {'Mean': np.mean, 'Standard Deviation': statistics.stdev}}).reset_index()
    faculty_ID.columns = faculty_ID.columns.droplevel()
    faculty_ID = faculty_ID.rename(columns={faculty_ID.columns[0]: "FACULTY_ID", faculty_ID.columns[1]: "MEAN", faculty_ID.columns[2]: "STANDARD_DEVIATION"})
    
    faculty_ID['difference'] = faculty_ID['MEAN'] - mean
    
    df['MEAN_DIFF'] = df.apply (lambda row: match_id_diff(faculty_ID, row['FACULTY_ID']), axis=1)
    return df
    
def match_id_diff(faculty_ID_df, faculty_id):
    return faculty_ID_df.loc[faculty_ID_df['FACULTY_ID'] == faculty_id, 'difference'].iloc[0]

def concordance_output_apt(df):
    df['APTITUDE_SCORE'] = df['APTITUDE_SCORE'].astype(float)
    df['APTITUDE_SCORE'] = np.where(df['APTITUDE_SCORE'] >= 4.0, 1,0)
    
    total = 0 # total number of obs
    concordant = 0 # concordant apt scores
    discordant = 0
    newDF = pd.DataFrame()
        
    df = df.groupby(['AMCAS_ID']).agg({'APP_YEAR': lambda col: col.tolist(),
                                 'FACULTY_ID': lambda col: col.tolist(),
                                 'MONTHS': lambda col: col.tolist(),
                                 'EXPERIENCE_DESCR': lambda col: col.tolist(),
                                 'TOTAL_HOURS': lambda col: col.tolist(),
                                 'APTITUDE_SCORE': lambda col: col.tolist()}).reset_index()
        
    for index, row in df.iterrows():
        total += 1
        tempAptitudeScore = list(set(row.APTITUDE_SCORE))
        if len(tempAptitudeScore) == 1:
            newDF = newDF.append(row)
            concordant += 1
        else: 
            discordant += 1 
    print ("concordant: ", (concordant/total), "discordant: ", (discordant/total))
    return newDF

def concordance_output_z(df, input_):
    # input_ == True, 
    # input_ == False, z score = output
 
    if input_:
        df = faculty_mean_diff(df)
        df['APTITUDE_SCORE'] = df['APTITUDE_SCORE'].astype(float)
        df['APTITUDE_SCORE'] = np.where(df['APTITUDE_SCORE'] >= 4.0, 1,0)
        df = df.groupby(['AMCAS_ID']).agg({'APP_YEAR': lambda col: col.tolist(),
                                 'FACULTY_ID': lambda col: col.tolist(),
                                 'MONTHS': lambda col: col.tolist(),
                                 'EXPERIENCE_DESCR': lambda col: col.tolist(),
                                 'TOTAL_HOURS': lambda col: col.tolist(),
                                 'APTITUDE_SCORE': lambda col: col.tolist(),
#                                  'Z_SCORE': lambda col: col.tolist(),
                                 'MEAN_DIFF': lambda col: col.tolist()}).reset_index()
    else:
        df = z_score(df)
        df['Z_SCORE'] = np.where(df['Z_SCORE'] >= 0.25, 1,0)
        df = df.groupby(['AMCAS_ID']).agg({'APP_YEAR': lambda col: col.tolist(),
                                 'FACULTY_ID': lambda col: col.tolist(),
                                 'MONTHS': lambda col: col.tolist(),
                                 'EXPERIENCE_DESCR': lambda col: col.tolist(),
                                 'TOTAL_HOURS': lambda col: col.tolist(),
                                 'APTITUDE_SCORE': lambda col: col.tolist(),
                                 'Z_SCORE': lambda col: col.tolist()}).reset_index()
        
    total = 0 # total number of obs
    concordant = 0 # concordant apt scores
    discordant = 0
    newDF = pd.DataFrame()
    
    twoFaculty = 0
    morethan2 = 0
    
    mean_diff1 = []
    mean_diff2 = []
    for index, row in df.iterrows():
        total += 1
        if input_:
            tempScore = list(set(row.APTITUDE_SCORE))
        else:
            tempScore = list(set(row.Z_SCORE))
        if len(tempScore) == 1:
            concordant += 1
            tempFacultyID = list(set(row.FACULTY_ID))
            if len(tempFacultyID) == 2:
                newDF = newDF.append(row)
                twoFaculty += 1
                if input_:
                    mean_diff1.append(row.MEAN_DIFF[0])
                    mean_diff2.append(row.MEAN_DIFF[1])
            else: 
                morethan2 += 1
        else: 
            discordant += 1 
    if input_:
        newDF['MEAN_DIFF1']=mean_diff1
        newDF['MEAN_DIFF2']=mean_diff2
    print("concordant & 2 faculty IDs: ", (twoFaculty/total))
    print ("concordant: ", (concordant/total), "discordant: ", (discordant/total))
    return newDF
    
def combine_exp(df, apt_score, input_ = False):
#     if (apt_score == False):
#         z_score_temp = df[['AMCAS_ID','Z_SCORE1','Z_SCORE2']]
# #         df = df.loc[:, df.columns != 'Z_SCORE']
#         df = df.loc[:, df.columns != 'Z_SCORE1']
#         df = df.loc[:, df.columns != 'Z_SCORE2']
    if input_ == True and apt_score == False:
        mean_diff = df[['AMCAS_ID','MEAN_DIFF1','MEAN_DIFF2']]
        df = df.loc[:, df.columns != 'MEAN_DIFF1']
        df = df.loc[:, df.columns != 'MEAN_DIFF2']
    UNIQUE_EXP = []
    MONTHS3 = []
    MONTHS6 = []
    MONTHS12 = []
    for index, row in df.iterrows():
        if ('ORGANIZATION_NAME' in df.columns):
            row.ORGANIZATION_NAME = list(set(row.ORGANIZATION_NAME))
            row.ORGANIZATION_NAME = [i.replace('[^a-zA-Z]',' ').lower() for i in row.ORGANIZATION_NAME] 
        
#         get latest application
        row.APP_YEAR = list(set(row.APP_YEAR))
        UNIQUE_EXP.append(len(row.EXPERIENCE_DESCR)/2)
        row.EXPERIENCE_DESCR = list(set(row.EXPERIENCE_DESCR))

        ind = row.APP_YEAR.index(max(row.APP_YEAR))
        if ind >= len(row.EXPERIENCE_DESCR):
            row.EXPERIENCE_DESCR = (row.EXPERIENCE_DESCR[-1])
        else:
            row.EXPERIENCE_DESCR = (row.EXPERIENCE_DESCR[ind])
        row.APP_YEAR = max(row.APP_YEAR)
        if (apt_score == False and input_ == False):
            row.Z_SCORE = float(sum(row.Z_SCORE)/len(row.Z_SCORE))
            
        row.APTITUDE_SCORE = float(sum(row.APTITUDE_SCORE)/len(row.APTITUDE_SCORE))
    
        row.TOTAL_HOURS = sum(row.TOTAL_HOURS)

        row.MONTHS = list(set(row.MONTHS))
        count3 = len([i for i in row.MONTHS if i >= 3 and i < 6]) 
        count6 = len([i for i in row.MONTHS if i >= 6 and i < 12]) 
        count12 = len([i for i in row.MONTHS if i >= 12]) 
        
        row.FACULTY_ID = list(set(row.FACULTY_ID))

        MONTHS3.append(count3)
        MONTHS6.append(count6)
        MONTHS12.append(count12)


    df['UNIQUE_EXP'] = UNIQUE_EXP
    df['MONTHS3'] = MONTHS3
    df['MONTHS6'] = MONTHS6
    df['MONTHS12'] = MONTHS12
    
#     if (apt_score == False):
#         df = pd.merge(df, z_score_temp)
    if (input_ == True  and apt_score == False):
        df = pd.merge(df, mean_diff)
   
    
    return df
    
def add_pub(df):
    df['AMCAS_ID']=df['AMCAS_ID'].astype(int)
    df2 = pd.read_csv("/gpfs/data/iime/data/admissions/non_urm_matrix.csv")
    df2 = df2[['aamc_id','exp_publications']]
    df2 = df2.rename(columns={"aamc_id": "AMCAS_ID", "exp_publications": "PUBLICATIONS"})
    df = pd.merge(df, df2)
    return df

def process_reg(og_df, columnName, apt_score = True):
    df = og_df.copy()
    df = replace_newline(df)
    df = process_months(df)
    df = df.groupby(['AMCAS_ID']).agg({'APP_YEAR': lambda col: col.tolist(),
                                 'ORGANIZATION_NAME': lambda col: col.tolist(),
                                 'FACULTY_ID': lambda col: col.tolist(),
                                 'MONTHS': lambda col: col.tolist(),
                                 'EXPERIENCE_DESCR': lambda col: col.tolist(),
                                 'TOTAL_HOURS': lambda col: col.tolist(),
                                 'APTITUDE_SCORE': lambda col: col.tolist()}).reset_index()
    df = combine_exp(df, apt_score)
    df['APTITUDE_SCORE'] = df['APTITUDE_SCORE'].astype(float)
    df['APTITUDE_SCORE'] = np.where(df['APTITUDE_SCORE'] >= 4.0, 1,0)
    df = add_pub(df)
    df[columnName] = df[columnName].str.replace('[^a-zA-Z]',' ').str.lower()
    df["TOTAL_HOURS"] = pd.to_numeric(df["TOTAL_HOURS"], downcast="float")
    df = df.dropna() 
    print("finished processing, ", len(df), " rows")
    print("1:0 ratio = ", len(df[df.APTITUDE_SCORE == 1]) / len(df) , ":", len(df[df.APTITUDE_SCORE == 0]) / len(df))
    
    return df
    
def process_all(og_df, columnName, apt_score = True, input_= True):
    df = og_df.copy()
    df = replace_newline(df)
    df = process_months(df)
    
    if apt_score:
        df = concordance_output_apt(df)
    else:
        df = concordance_output_z(df, input_)
        
    df = combine_exp(df, apt_score, input_)
    df = add_pub(df)
    df[columnName] = df[columnName].str.replace('[^a-zA-Z]',' ').str.lower()
    df["TOTAL_HOURS"] = pd.to_numeric(df["TOTAL_HOURS"], downcast="float")
    df = df.dropna() 
    if apt_score or input_:
        print("finished processing, ", len(df), " rows")
        print("1:0 ratio = ", len(df[df.APTITUDE_SCORE == 1.0]) / len(df) , ":", len(df[df.APTITUDE_SCORE == 0.0]) / len(df))
    else:
        print("finished processing, ", len(df), " rows")
        print("1:0 ratio = ", len(df[df.Z_SCORE >= 1.0]) / len(df) , ":", len(df[df.Z_SCORE == 0.0]) / len(df))
    
    return df

def process_zinput(og_df, columnName):
    df = og_df.copy()
    df = replace_newline(df)
    df = process_months(df)

def graph_aptitude_z_score(og_df, columnName, apt_score = False):
    df = og_df.copy()
    df = replace_newline(df)
    df = z_score(df)
    df = df.dropna() 
    print("finished processing, ", len(df), " rows")
    return df