import nltk
import re
import spacy
from pathlib import Path
import pandas as pd
from collections import Counter
from math import log
from nltk.corpus import cmudict

nlp = spacy.load('en_core_web_sm')
nlp.max_length = 2000000

d = cmudict.dict()

def read_novels(path : str):
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
df = read_novels(path)
# print(df.head())


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

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    word = text.split()
    word_count = len(word)
    sentence_count = text.count('.' and '?' and '!')
    syllable_count = sum(count_syl(w, d) for w in word)

    if word_count == 0 or sentence_count == 0:
        return 0.00
    else:
        fk_grade_level = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
        return fk_grade_level

# print(fk_level(df.iloc[0]['text'], d))
# print(fk_level('', d))
    

def parse(df, store_path=Path.cwd() / "syntax-style" / "novels" / "parsed", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    nlp = spacy.load("en_core_web_sm")

    store_path.mkdir(parents=True, exist_ok = True)
    nlp.max_length = 1200000
    
    parsed_texts = []
    for index, row in df.iterrows():
        print(f"Parsing text from row {index + 1} out of {len(df)}")
        parsed_text = nlp(row['text'])
        parsed_texts.append(parsed_text)

    df['parsed_text'] = parsed_texts
    df.to_pickle(store_path/out_name)
    return df

# parsed_df = parse(df)
# print(parsed_df.info())

def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = nltk.word_tokenize(text)
    ttr = len(set(tokens)) / len(tokens)
    return ttr


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
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
    
    subjects = []
    verb_counts = Counter()

    for sentence in doc.sents:
        for token in sentence:
            if token.lemma_ == target_verb:
                subjects.extend([child.text for child in token.children if child.dep_ == "nsubj"])
                verb_counts[target_verb] += 1
    
    subject_counts = Counter(subjects)
    total_subjects = sum(subject_counts.values())

    pmi_subjects = []
    for subject, count in subject_counts.items():
        pmi = log((count / total_subjects) / (verb_counts[target_verb] / len(doc)), 2)
        pmi_subjects.append((subject, pmi))

    return sorted(pmi_subjects, key=(lambda x: x[1]), reverse=True)


def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list of tuples."""
    subjects = []

    for sentence in doc.sents:
        for token in sentence:
            if token.lemma_ == verb:
                subjects.extend([child.text for child in token.children if child.dep_ == "nsubj"])

    subject_counts = Counter(subjects)
    return subject_counts.most_common()


def subject_counts(doc):
    """Extracts the most common subjects in a parsed document. Returns a list of tuples."""
    subjects = []

    for sentence in doc.sents:
        for token in sentence:
            if token.dep_ == "nsubj":
                subjects.append(token.text)

    subject_counts = Counter(subjects)
    return subject_counts.most_common()


def get_subjects(df):
    """Extracts the most common subjects from the parsed texts in the DataFrame. Returns dictionary"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = subject_counts(row["parsed_text"])
    return results

if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "syntax-style" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    nltk.download("cmudict")
    parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "syntax-style" / "novels" / "parsed" / "parsed.pickle")
    print(get_subjects(df))
     
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed_text"], "say"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed_text"], "say"))
        print("\n")
    

