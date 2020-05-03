import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer = WordNetLemmatizer()

titles = [line.rstrip() for line in open('./nlp_class/all_book_titles.txt')]

stopwords = set(w.rstrip() for w in open('./nlp_class/stopwords.txt'))

## an alternative source of stopwords is to directly import them from the
## nltk library

## Let's add more stopwords that are specific to this problem
## They don't really have much meaning in the context of this problem
stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth'
})

def my_tokenizer(s):
    s = s.lower()
    ## Splits string into words
    tokens = nltk.tokenize.word_tokenize(s)
    ## Remove short words, they're probably not useful
    tokens = [t for t in tokens if len(t) > 2]
    ## Put words into base form
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    ## remove the stopwords
    tokens = [t for t in tokens if t not in stopwords]
    ## remove any digits (i.e. 3rd edition)
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens

## Create a word to index map so that we can create our word-frequency vectors later
## let's also save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []
error_count = 0
for title in titles:
    try:
        ## Throw an exception if we have some bad characters
        title = title.encode('ascii', 'ignore').decode('utf-8')
        all_titles.append(title)
        tokens = my_tokenizer(title)
        all_tokens.append(tokens)
        ## Try to find a function to get the unique words like this
        ## without the complexity
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except Exception as e:
        print(e)
        print(title)
        error_count +=1


print('Number of errors parsing file: ', error_count, "number of lines in file: ", len(titles))
if error_count == len(titles):
    print('There is no data to do anything with! Quitting...')
    exit()

## Now let's create our input matrices
## Just indicator variables for this example
## Works better than proportions
def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] = 1
    return x

N = len(all_tokens)
D = len(word_index_map)
## Terms along rows, documents along columns
X = np.zeros((D,N))
i = 0
for tokens in all_tokens:
    X[:,i] = tokens_to_vector(tokens)
    i+= 1

def main():
    svd = TruncatedSVD()
    Z = svd.fit_transform(X)
    plt.scatter(Z[:,0], Z[:,1])
    for i in range(D):
        plt.annotate(s = index_word_map[i], xy=(Z[i,0], Z[i,1]))
    plt.show()

if __name__=='__main__':
    main()

