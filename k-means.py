import math
import string
import json
import random
from leveldb import LevelDB

class NewsRepo:
    db = None
    iter = None
    def __init__(self):
        self.db = LevelDB('/path')
        self.iter = self.db.RangeIter()

    def getNextDoc(self):
        try:
            url, doc = next(self.iter)
            url = url.decode('utf-8')
            doc = doc.decode('utf-8')

            d = json.loads(doc)
            d['url'] = url

            return d
        except Exception as e:
            print(str(e))
            return None

repo = NewsRepo()
doc = repo.getNextDoc()

def count_object(seq):
    """Calculate frequency of an object in a list.

    Args:
        seq: a list with strings as its elements.

    Returns:
        a dict whose keys are objects in a list
        and the corresponding values are the frequency of a key.
    """
    count_dict = {}
    for x in seq:
        if x not in count_dict:
            count_dict[x] = 1
        else:
            count_dict[x] += 1
    return count_dict

def remove_char(words_list):
    """Remove "'s" of a word from an array."""
    for i in range(len(words_list)):
        if words_list[i].endswith("'s"):
            words_list[i] = words_list[i][:-2]
    return words_list

def get_total_docs(doc):
    """Append all docs' contents into one list.

    Args:
        a doc is a dict containing a BBC news article, whose keys are 'intro',
        'time', 'content', 'url' and 'title'.
    """
    total_docs_content = []
    while doc is not None:
        total_docs_content.append(doc['content'])
        doc = repo.getNextDoc()
    return total_docs_content

def total_docs_split_word(total_docs):
    """split each doc's content word by word and calculate a word's frequency.

    Args:
        total_docs: an array containing contents of all docs.
        One element in the array is a doc's contents.

    Returns:
        a list containing tuples, each of which represents a doc.
        The first element in a tuple is a dict whose keys are words
        and values are frequencies of the corresponding words in a doc.
        The second element is the related doc number.
    """
    total_docs_split = []
    doc_num = 0
    for i in range(len(total_docs)):
        words_1 = total_docs[i].split()
        words_2 = [word.strip(string.punctuation).lower() for word in words_1]
        words_2 = remove_char(words_2)
        doc_words_dict = count_object(words_2)
        doc_num += 1
        total_docs_split.append((doc_words_dict, doc_num))
    return total_docs_split

def get_all_docs_words(total_docs_words):
    """calculate every word's frequency in all docs' contents.

    Args:
        total_docs_words: an array containing words from docs' contents
        in the form of tuples. In a tuple, the 1st element is a dict,
        whose keys are words from a doc and values are corresponding frequency
        of the word. The 2nd element of a tuple is the related doc number.

    Returns:
        a dict whose keys are words
        and values are numbers of how many docs a word occurs.
    """
    all_docs_words_dict = {}
    for i in range(len(total_docs_words)):
        doc_dict = total_docs_words[i][0]
        doc_set = set(doc_dict.keys())
        for word in doc_set:
            if word not in all_docs_words_dict:
                all_docs_words_dict[word] = 1
            else:
                all_docs_words_dict[word] += 1
    return all_docs_words_dict

def calculate_doc_size(one_doc_dict):
    """Sum the frequencies of all words in a doc's contents.

    Args:
        one_doc_dict: a dict representing one doc's contents,
        whose keys are words in a doc's contents
        and values are frequencies of the corresponding words.
    """
    doc_size = 0
    for word in one_doc_dict.keys():
        doc_size += one_doc_dict[word]
    return doc_size

def compute_tfidf(total_docs_words, all_docs_dict):
    """compute every word's tf*idf in a doc's contents.

    Args:
        total_docs_words: an array containing words from docs' contents
        in the form of tuples. In a tuple, the 1st element is a dict,
        whose keys are words from a doc and values are corresponding frequency
        of the word. The 2nd element of a tuple is the related doc number.

        all_docs_dict: a dict whose keys are words from all docs' contents
        and values are numbers of how many docs the related word occurs.

    Returns:
        a list embedded dicts. a dict represents a doc's contents,
        whose keys are words and values are the results of tf*idf
        of corresponding words.
    """
    total_tfidf = []
    total_docs_size = len(total_docs_words)

    for i in range(len(total_docs_words)):
        result_tfidf = {}
        doc_dict = total_docs_words[i][0]
        one_doc_size = calculate_doc_size(doc_dict)
        for word in doc_dict:
            if word == '':
                continue
            else:
                tf = doc_dict[word] / one_doc_size
                idf = math.log(total_docs_size / all_docs_dict[word])
                result_tfidf[word] = tf * idf
        total_tfidf.append(result_tfidf)
    return total_tfidf

def update_total_docs_list(total_docs_tfidf_list, total_docs_words_list):
    """create a new list containing words, their frequencies, tfidf and doc num.

    Args:
        total_docs_tfidf_list: a list containing dicts. The keys of a dict are
        words from one doc's contents and values are tfidf of the corresponding
        words.

        total_docs_words_list: a list embedded tuples. a tuple represents one doc.
        The 1st element of a tuple is a dict, whose keys are words from a doc's
        contents and values are frequencies of the corresponding words.
        The 2nd element is the doc num indicating which doc is.

    Returns:
        A list embedded tuples, each of which contains a dict as the 1st element.
        The keys of a dict are words from a doc's contents and its values are
        the frequency of the related word and its tf*idf in a form of tuple.
    """
    new_list = []
    doc_num = 0
    for i in range(len(total_docs_tfidf_list)):
        vector_doc = {}
        doc_num += 1
        document_dict = total_docs_tfidf_list[i]
        for word in document_dict:
            vector_doc[word] = (total_docs_words_list[i][0][word], document_dict[word])
        new_list.append((vector_doc,doc_num))
    return new_list

def sort_dict(D):
    """Sort a dict by the 2nd element of the values in a descent order."""
    return sorted(D.items(), key = lambda x: x[1], reverse = True)

def computer_avg_tfidf(total_docs_list):
    """Compute the mean values of tf-idf of words from all docs' contents.

    Returns:
        a dict whose keys are words and values are the avg values of (tf*idf)**2 of words.
    """
    D_sum = {}  # a key is a word and its value is the sum value of (tf*idf)**2 of the word
    D_avg = {}  # a key is a word and its value is the avg value of (tf*idf)**2 of the word
    total_docs_size = len(total_docs_list)
    for i in range(len(total_docs_list)):
        doc_dict = total_docs_list[i][0]
        for word in doc_dict:
            if word not in D_sum:
                D_sum[word] = (doc_dict[word][1]) ** 2
            else:
                D_sum[word] += (doc_dict[word][1]) ** 2
    for word in D_sum:
        D_avg[word] = D_sum[word]/total_docs_size
    return D_avg

def select_words(avg_tfidf, all_docs_dict):
    """Select 2000 words that occurred in at least 30 docs
    and whose tfidf values in top 2000.

    Args:
        avg_tfidf: a dict whose keys are words and values are avg values of tfidf.

    Returns:
        a list that contains words whose avg values of tfidf are in the top 2000.
    """
    select_words_dict = {}
    for word in avg_tfidf:
        if all_docs_dict[word] >= 30:    # a word which occurrs more than 30 docs
            select_words_dict[word] = avg_tfidf[word]
    select_2000_words = sort_dict(select_words_dict)[0:2001]

    select_words_list = []
    for i in range(len(select_2000_words)):
        select_words_list.append(select_2000_words[i][0])
    return select_words_list

def vector_standard(total_docs_list, select_2000_words_list):
    """Standardize every doc as a 2000-division vector.

    Args:
        total_docs_list: a list embedded tuples. one tuple represents a doc.
        The 1st element of a tuple is a dict, whose keys are words and values
        are a tuple containing the coressponding words' frequencies and tfidf.

        select_2000_words_list: a list containing 2000 words with the top tfidf.

    Returns:
        a list that contains vectors in the form of dicts.
        Each vector representing a doc's content has 2000 divisions.
    """
    vector_list = []
    for i in range(len(total_docs_list)):
        vector_dict = {}
        doc_dict = total_docs_list[i][0]
        doc_num = total_docs_list[i][1]
        for word in doc_dict:
            if word in select_2000_words_list:
                vector_dict[word] = doc_dict[word][1]
        vector_list.append((vector_dict, doc_num))
    return vector_list

def distance(vector1, vector2):
    """Compute Euclidean distance of two vectors in the form of two dicts.
    """
    result = 0
    for x in vector1:
        if x in vector2:
            result += (vector1[x] - vector2[x]) ** 2
        else:
            result += (vector1[x]) ** 2
    for y in vector2:
        if y not in vector1:
            result += (vector2[y]) ** 2
    return math.sqrt(result)

def assign_clusters(vector_list, centroids):
    """Assign a vector to its nearest centroid.
    Args:
        vector_list: a list containing vectors. One vector represents one doc.
        centroids: a list containing n vectors. each vector is a centroid.

    Returns:
        a clusters list embedded n lists. e.g. A vector with the nearest distance with the jth centroids,
        then this vector is assigned to the jth list in clusters.
    """
    rows = 10
    clusters = []
    for row in range(rows):
        clusters += [[]]

    for i in range(len(vector_list)):
        min_dist = 1000
        min_index = 0
        seen = False
        for j in range(len(centroids)):
            dist = distance(vector_list[i][0], centroids[j])
            if seen:
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            else:
                min_dist = dist
                seen = True
        clusters[min_index].append(vector_list[i])
    return clusters

def update_centroids(clusters):
    """Update centroids after assigning vectors to clusters.

    In one cluster, the average of vectors is a new centroid."""
    new_centroids_list = []
    for i in range(len(clusters)):
        sum_value = {}
        avg_value = {}
        length = len(clusters[i])
        for j in range(len(clusters[i])):
            clusters_dict = clusters[i][j][0]
            for word in clusters_dict:
                if word not in sum_value:
                    sum_value[word] = clusters_dict[word]
                else:
                    sum_value[word] += clusters_dict[word]
        for x in sum_value:
            avg_value[x] = sum_value[x]/length
        new_centroids_list.append(avg_value)
    return new_centroids_list

def main():

    total_docs_content_list = get_total_docs(doc)
    total_docs_words_list = total_docs_split_word(total_docs_content_list)
    all_docs_dict = get_all_docs_words(total_docs_words_list)
    total_docs_tfidf = compute_tfidf(total_docs_words_list, all_docs_dict)
    new_total_docs_list = update_total_docs_list(total_docs_tfidf, total_docs_words_list)
    avg_tfidf_dict = computer_avg_tfidf(new_total_docs_list)
    select_2000_words_list = select_words(avg_tfidf_dict, all_docs_dict)
    vector_standard_list = vector_standard(new_total_docs_list, select_2000_words_list)

    centroids_0 = []  # randomly initialize centroids
    k = 1
    while k <= 10:
        u = random.choice(vector_standard_list)[0]
        centroids_0.append(u)
        k += 1

    centroids = centroids_0
    count = 0
    while count < 1000:
        clusters = assign_clusters(vector_standard_list, centroids)
        count += 1
        if count % 20 == 0:
            clusters_s = json.dumps(clusters)
            s = "/path/clusters" + str(count)
            with open(s, 'w') as f:
                f.write(clusters_s)
        print(count)
        centroids = update_centroids(clusters)

if __name__ == '__main__':
    main()
