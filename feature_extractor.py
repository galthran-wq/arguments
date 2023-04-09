#!/usr/bin/python
# -*- coding: utf-8 -*-
####
#
# Some features based on Tobias Milz's ArguE project.
# See more: https://github.com/Milzi/ArguE
#
####

import nltk
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import os
import inspect

current_dir = os.path.dirname(inspect.stack()[0][1])

def get_propositions(dataset, column='arg1',
                     tokenizer=nltk.tokenize.word_tokenize,
                     only_props=False):
    """Parse propositions
    dataset: the original dataframe
    tokenizer: nltk.tokenize.word_tokenize, bert-embedding.tokenizer, etc.
    Output:
        propositionSet: list of the propositions of the arg1
        parsedPropositions: parsed prop. in the arg1
    """

    propositionSet = list(set(dataset[column]))

    if column[-1] == '1':
        column2 = column[:-1] + '2'
        if column2 in dataset.keys():
            propSet2 = list(set(dataset[column2]))
            propositionSet = propositionSet + propSet2

    propositionSet = list(dict.fromkeys(propositionSet))
    parsedPropositions = list()

    if only_props:
        return (propositionSet, parsedPropositions)

    for proposition in propositionSet:
        words = tokenizer(proposition)
        parsedPropositions.append(nltk.pos_tag(words))

    return (propositionSet, parsedPropositions)

def _pad_sequence(list, maxlen):
    """
    list: list of matrices (sequence_length, pos_tag_one_hot_vector)
    return padded 3d array
    """
    max_length = max([row.shape[0] for row in list])
    list = [ np.pad(m, [(0, max_length - m.shape[0]), (0, 0)]) for m in list ]
    list = np.array(list)[:, :maxlen, :]
    return list


def add_pos_feature(dataset,
                    propositionSet,
                    parsedPropositions,
                    pad_no=35,
                    has_2=True,
                    ):
    """Add Part-of-Speech features for every proposition"""

    tagdict = nltk.data.load('help/tagsets/upenn_tagset.pickle')
    lb = LabelBinarizer()
    lb.fit(list(tagdict.keys()))

    propositionPOSList = list()

    current = 0
    for proposition in parsedPropositions:

        propositionPOS = get_one_hot_pos(proposition, lb)
        propositionPOSList.append(propositionPOS)

    propositionPOSPadded = _pad_sequence(propositionPOSList,
                                            maxlen=pad_no)

    posFrame = pd.DataFrame({'arg1': propositionSet,
                            'pos1': propositionPOSPadded.reshape(propositionPOSPadded.shape[0], -1).tolist()})
    dataset = pd.merge(dataset, posFrame, on='arg1')
    if has_2:
        posFrame = posFrame.rename(columns={'arg1': 'arg2',
                                   'pos1': 'pos2'})
        dataset = pd.merge(dataset, posFrame, on='arg2')

    return dataset


def get_one_hot_pos(parsedProposition, label_binarizer):
    """Get one-hot encoded PoS for the proposition"""

    posVectorList = label_binarizer.transform([word[1] for word in
                                              parsedProposition])
    posVector = np.array(posVectorList)

    return posVector


def read_key_words(file):
    """Reads list of words in file, one keyword per line"""

    return [line.rstrip('\n') for line in open(file)]


def add_token_feature(dataset,
                      propositionSet,
                      parsedPropositions,
                      has_2=True,
                      ):
    """Add number of propositions in the arguments of the dataset"""

    numberOfTokens = list()

    for i in range(len(propositionSet)):

        numberOfTokens.append([propositionSet[i],
                              len(parsedPropositions[i])])

    tokenDataFrame = pd.DataFrame(data=numberOfTokens,
                                  columns=['proposition', 'tokens'])

    tokenDataFrame = \
        tokenDataFrame.rename(columns={'proposition': 'arg1',
                              'tokens': 'tokensArg1'})

    dataset = pd.merge(dataset, tokenDataFrame, on='arg1')
    if has_2:
        tokenDataFrame = tokenDataFrame.rename(columns={
                                        'arg1': 'arg2',
                                        'tokensArg1': 'tokensArg2'})
        dataset = pd.merge(dataset, tokenDataFrame, on='arg2')

    return dataset


def add_shared_words_feature(dataset,
                             propositionSet,
                             parsedPropositions,
                             key='arg',
                             word_type='nouns',
                             min_word_length=0,
                             stemming=False,
                             fullText=False,
                             fullPropositionSet=None,
                             fullParsedPropositions=None,
                             has_2=True,
                             ):
    """Add binary has shared noun and number of shared nouns to the dataset"""

    if not has_2 and not fullText:
        return dataset
    full = ''
    if fullText:
        full = 'Full'

    if stemming:
        ps = nltk.stem.PorterStemmer()
        stemmed = 'Stem'
    else:
        ps = None
        stemmed = ''
    key1 = key + '1'
    key2 = key + '2'
    word_key = word_type.title()
    if key == 'arg':
        ret_keys = 'shared' + stemmed + word_key + full
        ret_keyn = 'numberOfShared' + stemmed + word_key + full
    else:
        ret_keys = 'originalShared' + stemmed + word_key + full
        ret_keyn = 'originalNumberOfShared' + stemmed + word_key + full

    if word_type == 'nouns':
        pos_tag_list = ['NN']
    else:
        if word_type == 'verbs':
            pos_tag_list = ['VB']
        else:
            pos_tag_list = []
    if not fullText:
        temp = dataset[[key1, key2]
                       ].apply(lambda row:
                               find_shared_words(parsedPropositions[
                                                    propositionSet.index(row[
                                                        key1])],
                                                 parsedPropositions[
                                                     propositionSet.index(row[
                                                         key2])],
                                                 min_length=min_word_length,
                                                 pos_tag_list=pos_tag_list,
                                                 stemming=stemming,
                                                 ps=ps,
                                                 ), axis=1)
        temp = pd.DataFrame(temp.tolist(), columns=['sharedNouns',
                            'numberOfSharedNouns'])
        dataset[ret_keys] = temp.loc[:, 'sharedNouns']
        dataset[ret_keyn] = temp.loc[:, 'numberOfSharedNouns']
    else:
        temp = dataset[[key1, 'fullText1']
                       ].apply(lambda row:
                               find_shared_words(
                                   parsedPropositions[
                                       propositionSet.index(row[key1])],
                                   fullParsedPropositions[
                                       fullPropositionSet.index(
                                           row['fullText1'])],
                                   min_length=min_word_length,
                                   pos_tag_list=pos_tag_list,
                                   stemming=stemming,
                                   ps=ps,
                                   ), axis=1)
        temp = pd.DataFrame(temp.tolist(), columns=['sharedNouns',
                            'numberOfSharedNouns'])
        dataset[ret_keys + '1'] = temp.loc[:, 'sharedNouns']
        dataset[ret_keyn + '1'] = temp.loc[:, 'numberOfSharedNouns']
        if has_2:
            temp = dataset[[key2, 'fullText1']
                           ].apply(lambda row:
                                   find_shared_words(
                                       parsedPropositions[
                                           propositionSet.index(row[key2])],
                                       fullParsedPropositions[
                                           fullPropositionSet.index(
                                               row['fullText1'])],
                                       min_length=min_word_length,
                                       pos_tag_list=pos_tag_list,
                                       stemming=stemming,
                                       ps=ps,
                                       ), axis=1)
            temp = pd.DataFrame(temp.tolist(), columns=['sharedNouns',
                                'numberOfSharedNouns'])
            dataset[ret_keys + '2'] = temp.loc[:, 'sharedNouns']
            dataset[ret_keyn + '2'] = temp.loc[:, 'numberOfSharedNouns']

    return dataset


def find_shared_words(proposition,
                      partner,
                      min_length=0,
                      pos_tag_list=['NN'],
                      stemming=False,
                      ps=None,
                      ):
    """Find shared words between prop and partner
    Input:
        proposition: search key
        partner: search target
        min_length: minimum length of the shared words
        pos_tag_list: PoS tag for collected words, [] for all
        stemming: True for using stemming
        ps: PorterStemmer
    Output:
        sharedWords: binary
        noSharedWords: number of shared words
    """

    has_tag_list = len(pos_tag_list) > 0
    if not stemming:
        arg1Nouns = [word for (word, pos) in proposition
                     if (not has_tag_list or
                         pos in pos_tag_list) and
                     len(word) >= min_length]
        arg2Nouns = [word for (word, pos) in partner
                     if (not has_tag_list or
                         pos in pos_tag_list) and
                     len(word) >= min_length]
    else:
        arg1Nouns = [ps.stem(word) for (word, pos) in proposition
                     if len(word) >= min_length]
        arg2Nouns = [ps.stem(word) for (word, pos) in partner
                     if len(word) >= min_length]

    intersection = set(arg1Nouns).intersection(arg2Nouns)
    shared = 0

    if len(intersection) > 0:
        shared = 1
        return [shared, len(intersection)]
    else:
        return [0.0, 0.0]


def add_same_sentence_feature(dataset, has_2=True):
    """Add binary feature true if the two
       argument has the same original sentence"""

    if not has_2:
        return dataset
    dataset['sameSentence'] = dataset[['originalArg1', 'arg2']
                                      ].apply(lambda row: int(bool(row['arg2']
                                              in row['originalArg1'])), axis=1)

    return dataset
