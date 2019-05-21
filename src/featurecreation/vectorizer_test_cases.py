import src.featurecreation.vectorizer as vc

# TEST CASES FOR TF-IDF CREATOR - run from root dir

# Case 1: vanilla run with list input
testDoc = ["Hello World",
           "Hello World! How are you today!",
           "Hello World! Why won't you talk back to me?",
           "Hello World! Why can't I say anything else apart from Hello World?!?!"]

tstVct = vc.FreqVectorizer(testDoc, (1, 1), 5000, vect_type = "tf-idf")
print(tstVct.freq_vect.shape)
for i, val in enumerate(tstVct.vectorizer.get_feature_names()):
    print("{}: {}".format(i, val))
# print(tstVct.tf_idf)

# Case 2: bigram run with list input

tstVct = vc.FreqVectorizer(testDoc, (1, 2), 5000, vect_type = "tf-idf")
print(tstVct.freq_vect.shape)
for i, val in enumerate(tstVct.vectorizer.get_feature_names()):
    print("{}: {}".format(i, val))
# print(tstVct.tf_idf)

# Case 3: Limited Vocab run with list input
tstVct = vc.FreqVectorizer(testDoc, (1, 2), 10, vect_type = "tf-idf")
for i, val in enumerate(tstVct.vectorizer.get_feature_names()):
    print("{}: {}".format(i, val))
print(tstVct.freq_vect.shape)
# print(tstVct.tf_idf)

# Case 4: Checking inheritance from torch data utils item
tstVct = vc.FreqVectorizer(testDoc, (1, 1), 5000, vect_type = "tf-idf")
print(tstVct[2])

# TEST CASES FOR BOW CREATOR - run from root dir

# Case 1: vanilla run with list input
testDoc = ["Hello World",
           "Hello World! How are you today!",
           "Hello World! Why won't you talk back to me?",
           "Hello World! Why can't I say anything else apart from Hello World?!?!"]

tstVct = vc.FreqVectorizer(testDoc, (1, 1), 5000, vect_type = "bow")
print(tstVct.freq_vect.shape)
for i, val in enumerate(tstVct.vectorizer.get_feature_names()):
    print("{}: {}".format(i, val))
# print(tstVct.tf_idf)

# Case 2: bigram run with list input

tstVct = vc.FreqVectorizer(testDoc, (1, 2), 5000, vect_type = "bow")
print(tstVct.freq_vect.shape)
for i, val in enumerate(tstVct.vectorizer.get_feature_names()):
    print("{}: {}".format(i, val))
# print(tstVct.tf_idf)

# Case 3: Limited Vocab run with list input
tstVct = vc.FreqVectorizer(testDoc, (1, 2), 10, vect_type = "bow")
for i, val in enumerate(tstVct.vectorizer.get_feature_names()):
    print("{}: {}".format(i, val))
print(tstVct.freq_vect.shape)
# print(tstVct.tf_idf)

# Case 4: Checking inheritance from torch data utils item
tstVct = vc.FreqVectorizer(testDoc, (1, 1), 5000, vect_type = "bow")
print(tstVct[2])