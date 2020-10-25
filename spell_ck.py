from itertools import groupby
import re
from nltk.tokenize import word_tokenize
import nltk
import pkg_resources
from symspellpy import SymSpell
nltk.download('punkt')

class spelling():
    def __init__(self, edit_distance = 3):
        self.edit_distance = edit_distance

        self.sym_spell = SymSpell(max_dictionary_edit_distance=self.edit_distance, prefix_length=7)

        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt")
        bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt")

        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)


    def rep_chars(self, text):
        ''' If repeated characters exceed count of 2, replace with 1 same character '''
        word = []
        for k, g in groupby(text):
            ti = sum(1 for _ in g)
            word.append(''.join(k*ti if ti < 3 else k))
        return ''.join(wd for wd in word)


    def check(self, text):
        '''
        -> tokenize
        -> make the first word lowercase (because I want to do it)
        -> (
            - first letter capital - append with a space
            - special character - just append
            - else: remove repeating characters and then spell correct
        )
        -> make the first word uppercase (capitalize)
        -> return output
        '''
        text = word_tokenize(text)
        text[0] = text[0].lower()

        final = []
        special = re.compile('[.,@_!#$%^&*()<>?/\|}{~:]')

        for word in text:
            if word.islower() or word.isupper(): # Leave capitalized words untouched # pre-processing
                word = self.rep_chars(word)
                word = self.sym_spell.lookup_compound(word, max_edit_distance=self.edit_distance, transfer_casing=True, ignore_non_words=True)
                word = word[0].term

            final.append(word)

        final[0] = final[0].capitalize()
        del text
        del word

        text = ''.join(word if special.search(word) != None else ' '+word for word in final)

        return text[1:]

