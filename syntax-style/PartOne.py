import nltk
import re
import spacy
from pathlib import Path
import pandas as pd
from collections import Counter
from nltk.corpus import cmudict

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

nltk.download('cmudict')
d = cmudict.dict()

def read_texts(path : str):
    """Checks file path is a directory and reads files in directory. If a file is a .txt file, the file is split and read.
    A pandas DataFrame is created using text, title, author and year from each .txt file. 
    Returns constructed pandas DataFrame.

    Args:
        path (str): directory containing .txt files

    Returns:
        df: constructed pandas DataFrame
    """
    data = []
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory.")

    for file in path.glob("*.txt"):
        title, author, year = file.stem.split('-')
        year = int(year) 

        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()

        data.append([text, title, author, year])

    df = pd.DataFrame(data, columns=['text', 'title', 'author', 'year'])
    df['year'] = df['year'].astype(int)
    df = df.sort_values(by='year', ignore_index=True)
    return df

path = Path.cwd()/"syntax-style/novels"
df = read_texts(path)
# print(df.head())

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    pronunciations = d.get(word.lower(), [])
    if pronunciations:
        return max(len(i) for i in pronunciations)
    else: 
        syllables = re.findall(r'[aeiouy]+', word.lower())
        return len(syllables)
    
# print(count_syl('Hello', d))
# print(count_syl('Love', d))

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    data = []
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory.")
    
    for file in path.glob("*.txt"):
        title, author, year = file.stem.split('-')
        year = int(year) 

        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()

        data.append([text, title, author, year])

    df = pd.DataFrame(data, columns=['text', 'title', 'author', 'year'])
    df['year'] = df['year'].astype(int)
    df = df.sort_values(by='year', ignore_index=True)
    return df

    


def parse(df, store_path=Path.cwd() / "texts" / "novels" / "parsed", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""

    pass


def regex_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using a regular expression."""
    tokens = re.findall(r'\b\w+\b', text)
    ttr = len(set(tokens)) / len(tokens)
    return ttr


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = nltk.word_tokenize(text)
    ttr = len(set(tokens)) / len(tokens)
    return ttr


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = regex_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list of tuples."""
    
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list of tuples."""
    pass



def subject_counts(doc):
    """Extracts the most common subjects in a parsed document. Returns a list of tuples."""
    pass





if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    #path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "texts" / "novels" / "parsed" / "name.pickle")
    # print(get_subjects(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "say"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "say"))
        print("\n")
    """

