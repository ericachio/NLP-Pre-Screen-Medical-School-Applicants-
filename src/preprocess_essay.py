import pandas as pd
import numpy as np
import textstat

def process_all(ogdf, columnName):
    df = ogdf.copy()
    
    df['lexicon_count'] = df[columnName].apply(textstat.lexicon_count)
    df['sentence_count'] = df[columnName].apply(textstat.sentence_count)
    df['syllable_count'] = df[columnName].apply(textstat.syllable_count)
    df['flesch_reading_ease'] = df[columnName].apply(textstat.flesch_reading_ease)
#     df['linsear_write_formula'] = df[columnName].apply(textstat.linsear_write_formula)
#     df['dale_chall_readability_score'] = df[columnName].apply(textstat.dale_chall_readability_score)
#     df['automated_readability_index'] = df[columnName].apply(textstat.automated_readability_index)
#     df['gunning_fog'] = df[columnName].apply(textstat.gunning_fog)
    print("half way done processing")
    df['standard_score'] = df[columnName].apply(lambda x: textstat.text_standard(x, float_output=True))
    df[columnName] = df[columnName].str.replace('[^a-zA-Z]',' ').str.lower()
    df['nyu_mentions'] = df[columnName].apply(lambda x: countNYU(x))
    df['lexical_density'] = df[columnName].apply(lambda x: lexical_density(x))
    
    cols_to_drop=['scrn_score_1','scrn_score_2', 'app_year']
    df = pd.DataFrame(df).drop(columns=cols_to_drop)
    
    df = df.dropna()
    
    print("finished processing")
    return df

def countNYU(string):
    text = string.split()
    count = text.count("nyu")
    count += text.count("new york university")
    return count

def lexical_density(text):
    return len(set(text)) / len(text)