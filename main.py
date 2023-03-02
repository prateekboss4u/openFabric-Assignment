import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
import spacy
import nltk
from nltk.corpus import wordnet
from spacy.matcher import Matcher
from spacy.tokens import Span

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################

# Download the wordnet corpus for nltk
nltk.download('wordnet')

# Load the small English model for spaCy
nlp = spacy.load('en_core_web_sm')

def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        response = get_response(text)
        output.append(response)
    return SimpleText(dict(text=output))

def get_response(question):
    # Tokenize the question using spaCy
    doc = nlp(question)

    # Initialize a list of possible answers
    answers = []

    # Define the patterns to match against
    pattern_1 = [{'LOWER': 'what'}, {'LOWER': 'is'}, {'POS': 'ADJ'}, {'POS': 'NOUN'}]
    pattern_2 = [{'LOWER': 'what'}, {'LOWER': 'is'}, {'POS': 'NOUN'}, {'POS': 'VERB'}]
    pattern_3 = [{'LOWER': 'what'}, {'LOWER': 'does'}, {'POS': 'PROPN'}, {'POS': 'VERB'}]

    # Initialize the spaCy matcher
    matcher = Matcher(nlp.vocab)

    # Add the patterns to the matcher
    matcher.add('pattern_1', None, pattern_1)
    matcher.add('pattern_2', None, pattern_2)
    matcher.add('pattern_3', None, pattern_3)

    # Find matches in the question
    matches = matcher(doc)

    # Iterate over the matches
    for match_id, start, end in matches:
        # Extract the matched span
        matched_span = doc[start:end]

        # Find the hypernyms of the matched span
        hypernyms = get_hypernyms(matched_span)

        # If we found any hypernyms, add them to the list of answers
        if hypernyms:
            answer = f"{matched_span.text} refers to {', '.join(hypernyms)}"
        else:
            answer = "I'm sorry, I don't know the answer."

        # Add the answer to the list of possible answers
        answers.append(answer)

    # If we found any possible answers, return the first one
    if answers:
        return answers[0]
    else:
        return "I'm sorry, I don't know the answer."

def get_hypernyms(span):
    # Initialize a list of hypernyms
    hypernyms = []

    # Iterate over the synsets of the span
    for synset in wordnet.synsets(span.text):
        # Get the hypernyms of the synset
        synset_hypernyms = synset.hypernyms()

        # Iterate over the synset hypernyms
        for synset_hypernym in synset_hypernyms:
            # Get the lemmas of the synset hypernym
            synset_hypernym_lemmas = synset_hypernym.lemmas()

            # Iterate over the synset hypernym lemmas
            for synset_hypernym_lemma in synset_hypernym_lemmas:
                # Add the synset hypernym lemma to the list of hypernyms
                hypernym = synset_hypernym_lemma.name().replace('_', ' ')
                hypernyms.append(hypernym)

    # Remove duplicates from the list of hypernyms
    hypernyms = list(set(hypernyms))

    return hypernyms

