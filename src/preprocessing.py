import numpy as np
import pandas as pd

def add_grades(X):
    """
    Compute the total number of As, Bs, Cs, etc for all different courses relevant
    to the admissions' committee (physics, chemistry).
    By default it drops the individual columns (e.g. Physics A/B, etc)
    Parameters:
    X (pandas.DataFrame)
    Returns:
    pandas.DataFrame: Updated input table with total grades
    """
    tmp_x = X.copy()
    suf = ('_A', '_B', '_C' ,'_D', '_F')

    for s in suf:
        new_col = 'total' + s + 's'
        full_cols = tmp_x.columns
        sufcol = full_cols.str.endswith(s)
        tmp_x[new_col] = tmp_x.loc[:,sufcol].sum(1)
    return tmp_x.drop(columns = full_cols[full_cols.str.endswith(suf)])

def add_region(X):
    """
    Group residency state at the regional level. Canada provinces are grouped as a
    country. US territories are grouped under Other. Residency column is dropped
    on output.

    The dictionary with states can be updated or eventually replaced by user input.

    Parameters:
    X (pandas.DataFrame)
    Returns:
    pandas.DataFrame: Updated input table with new region column.

    """
    tmp_x = X.copy()
    residency = tmp_x.residency_state.copy()
    state_dict = {
    'mid_atlantic': ['NY', 'NJ', 'PA', 'MD', 'DC', 'DE'],
    'mid_west': ['IL','IN','IA','KS','MI','MN','MO', 'NB', 'NE','ND','SD','OH','WI'],
    'ne_region': ['CT', 'RI', 'MA', 'NH', 'VT', 'ME'],
    'south': ['AL','AR','FL','GA','KY','LA','MS','NC', 'SC', 'TN','VA','WV'],
    'west': ['AK', 'CO', 'CA', 'HI', 'ID', 'MT', 'NV', 'OR', 'UT','WA', 'WY'],
    'west_south': ['TX', 'NM', 'AZ', 'OK'],
    'can': ['BC', 'AB', 'ON', 'QC', 'SK','MB','NS'],
    'other_us_missing': ['UC','VI', 'NL','PR', 'AP', 'AE', np.nan],
    }

    region_dict = {}
    for key, value in state_dict.items():
        for string in value:
            region_dict[string] = key

    tmp_x['region'] = residency.replace(region_dict)
    return tmp_x.drop(columns = 'residency_state')

def categorize_columns(X):
    """
    Converts objects and string columns into numbers (integers).

    Parameters:
    X (pandas.DataFrame)
    Returns:
    pandas.DataFrame: Updated input table with no string columns


    """

    tmp_x = X.copy()
    tmp_x['female'] = np.where(tmp_x.gender == "F", 1, 0)
    cat_cols = tmp_x.select_dtypes(include = ['object', 'category']).columns

    tmp_x[cat_cols] = tmp_x[cat_cols].astype('category').apply(lambda x: x.cat.codes)
    return tmp_x.drop(columns = 'gender')

def hours_to_years(X):
    """
    Scale down the hours of experience to years. Preferred method for scaling than
    normalizing with any other conventional approach (z-scores, min-max).

    Parameters:
    X (pandas.DataFrame)
    Returns:
    pandas.DataFrame: Updated input table with experiences in years

    """


    hrs_week = 35
    weeks_year = 52
    tmp_x = X.copy()
    cols = tmp_x.columns
    hrs_cols = cols[cols.str.contains('hours')]
    exp_names = hrs_cols.str.replace('hours', 'years')
    tmp_x[hrs_cols] = tmp_x[hrs_cols].apply(lambda x: x/hrs_week/weeks_year)
    tmp_x.rename(columns = dict(zip(hrs_cols.tolist(), exp_names.tolist())), inplace = True)
    return tmp_x

def log_income(X):
    """
    Scale down the income to log-scale. Preferred method for scaling than
    normalizing with any other conventional approach (z-scores, min-max).

    Parameters:
    X (pandas.DataFrame)
    Returns:
    pandas.DataFrame: Updated input table with log_income

    """

    tmp_x = X.copy()
    tmp_x['log_median_census_income'] = np.log(tmp_x['median_census_income'])
    return tmp_x.drop(columns = 'median_census_income')


def prepare_data(X):
    """
    # temporary solution
    # TODO: work with DB to pull all data into one source

    Join all preprocessing steps into a single function.

    Parameters:
    X (pandas.DataFrame)
    Returns:
    pandas.DataFrame: Updated input table with log_income

    """

    tmp_x = X.copy().drop(columns = 'outcome')
    tmp_x = add_grades(tmp_x)
    tmp_x = add_region(tmp_x)
    tmp_x = hours_to_years(tmp_x)
    tmp_x = log_income(tmp_x)
    tmp_x = categorize_columns(tmp_x)
    return tmp_x

# if __name__ == '__main__':
#     non_urm_matrix = pd.read_csv('data/raw/non_urm_matrix.csv',
#                         index_col=['aamc_id', 'application_year'])
#     print(prepare_data(non_urm_matrix).head())
