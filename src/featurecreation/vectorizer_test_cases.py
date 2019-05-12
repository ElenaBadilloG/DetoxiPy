# TEST CASES FOR TF-IDF CREATOR - run from root dir

import src.featurecreation.vectorizer as vc

# Case 1: vanilla run with list input
testDoc = ["Hello World",
           "Hello World! How are you today!",
           "Hello World! Why won't you talk back to me?",
           "Hello World! Why can't I say anything else apart from Hello World?!?!"]

tstVct = vc.TfidfData(testDoc, (1, 1), 5000)
print(tstVct.tf_idf.shape)
for i, val in enumerate(tstVct.vectorizer.get_feature_names()):
    print("{}: {}".format(i, val))
# print(tstVct.tf_idf)

# Case 2: bigram run with list input

tstVct = vc.TfidfData(testDoc, (1, 2), 5000)
print(tstVct.tf_idf.shape)
for i, val in enumerate(tstVct.vectorizer.get_feature_names()):
    print("{}: {}".format(i, val))
# print(tstVct.tf_idf)

# Case 3: Limited Vocab run with list input
tstVct = vc.TfidfData(testDoc, (1, 2), 10)
for i, val in enumerate(tstVct.vectorizer.get_feature_names()):
    print("{}: {}".format(i, val))
print(tstVct.tf_idf.shape)
# print(tstVct.tf_idf)

# Case 4: Checking inheritance from torch data utils item
tstVct = vc.TfidfData(testDoc, (1, 1), 5000)
print(tstVct[1])