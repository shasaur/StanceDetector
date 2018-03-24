from sentiwordnet_lexicon import SentiWordNetLexicon


#
# sent = SentiSynset()

# synonym = getSynonymForm('breakdown')
# print(synonym)

# print(list(swn.senti_synsets('slow')))
# true_synonym = getSynonymForm('slow')

#print(swn.senti_synset(synonym.))

#
# # def calculate_sentiwordnet_score(self, words):
#
# word_details = swn.senti_synset('breakdown.n.03')
# print(word_details)

lexer = SentiWordNetLexicon()

print(lexer.getSentimentOfWord('suck'))