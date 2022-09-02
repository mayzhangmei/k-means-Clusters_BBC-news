# k-means-Clusters_BBC-news

Dataset: 44407 BBC news articles

Goal: divide articles into 10 groups (clusters)

Method: K-Means

One article represents a doc. In a form of a dict, the ‘intro’, ‘time’, ‘content’, ‘url’ and ‘title’ of a doc are keys of the dict.

Step 1: clean the data

Split each doc’s contents word by word, and remove punctuations and special characters such as “’s”. 

For each doc in a form of a tuple, it contains two elements—the 1st one is a dict, whose keys are words from the doc’s contents and values are frequencies of the corresponding words; 
the 2nd one is the doc’s number.

Append all tuples into a list, that is “total_docs_words_list”.

Step 2: calculate tf*idf  and select words

Compute each word’s tf*idf from a doc’s content. In a form of a dict, keys are words and values are their tf*idf of corresponding words. 

Put all dicts into a list, that is “total_docs_tfidf”.

Compute average values of tf*idf of words from all docs’ contents, and select 2000 words with the top avg values of  tf*idf. 

Step 3: standardize vectors by using selected words

One doc’s content in a form of a dict, is considered as a vector. If selected words exist in a original doc’s content, keys of a dict are those existent words and values are the related words’  tf*idf. If not, the values of the non-existent words (keys) are 0.

Therefore, there are 44407 dicts considered as vectors, whose keys are selected words. Every vector has 2000 dimensions. 

Step 4: implement k-means clusters

Consider 10 centroids and randomly choose 10 vectors as initialized centroids. 

(1) Assign clusters: Calculate Euclidean distance between a vector and each of centroids. Find which centroid has the nearest distance with the vector, then assign this vector to the related cluster. E.g. if a vector is closest to the Nth centroids, and assign this vector to the Nth cluster.

(2) Update new centroids: In each cluster, the average values of vectors form a new centroid. 

Iterate above two procedures for many times.
