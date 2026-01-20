import nltk

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

text = "Nltk is a powerful NLP library! IT supports tokenizetion , stemming, and lemmatization. Let's test it :) #AI"
print("\nORIGINAL TEXT: ")
print(text)

# TOKENIZATION

from nltk.tokenize import WhitespaceTokenizer
wt = WhitespaceTokenizer()

print("\n1. WHITESPACE TOKENIZATION: ")
print(wt.tokenize(text))

from nltk.tokenize import wordpunct_tokenize

print("\n2. PUNCTUATION-BASED TOKENIZATION: ")
print(wordpunct_tokenize(text))

from nltk.tokenize import TreebankWordTokenizer
tbt= TreebankWordTokenizer()

print("\n3. TREEBANK TOKENIZATION: ")
print(tbt.tokenize(text))

from nltk.tokenize import TweetTokenizer
tt= TweetTokenizer()

print("\n3. TWEET TOKENIZER: ")
print(tt.tokenize(text))

from nltk.tokenize import MWETokenizer

sentence = "I love machine learning and live in New York"
mwe = MWETokenizer([('machine' , 'learning'), ('New', 'York')], separator='_')

print("\n5. MWE TOKENIZATION: ")
print(mwe.tokenize(sentence.split()))

# STEMMING

words = ["running", "files", "easily", "fairly", "studies", "studying"]

from nltk.stem import PorterStemmer 
ps = PorterStemmer()

print("\n6. PORTER STEMMER OUTPUT: ")
for word in words :
    print(word, " -> ", ps.stem(word))
    
#LEMMATIZATION

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

lemma_words = ["running", "better", "flies", "cars", "studies"]

print("\n8. LEMMATIZATION OUTPUT: ")
for word in lemma_words:
    print(word, " -> ", lemmatizer.lemmatize(word))
    

    
