import pandas as pd

def keywords():
    keywords = pd.read_excel(
        "C:/Users/Anna/Documents/GitHub/Bachelorarbeit/BA_Anna_ResultingComments/BA_Anna_ResultingComments.xlsx",
        sheet_name=0, index_col=0, usecols="A,B", skiprows=0)
    keywords = keywords.T
    return keywords

def comments():
    comments = pd.read_excel(
        "C:/Users/Anna/Documents/GitHub/Bachelorarbeit/BA_Anna_ResultingComments/BA_Anna_ResultingComments.xlsx",
        sheet_name=1, header=1)
    for i in comments.columns:
        comments[i].dropna(inplace=True)
    return comments