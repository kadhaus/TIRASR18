#!/usr/bin/env python
"""
    This program based on program, written by Martin Potthast
    from Weimar University.
    Program using the ChatNoir search engine to retrieve
    source for suspicious documents.
"""
# SourceRetrievalKres18, kres18sr_2.0

__author__ = 'Alexey Kreskin'
__email__ = 'leshakreskin@gmail.com'
__version__ = '2.0'

import codecs
import glob
import os
import simplejson
import sys
import time
import unicodedata
import re
#import urllib.request as urllib2
import urllib2

from cucco import Cucco
from gensim.models import KeyedVectors as kv
import numpy as np


class Logger:
    """
    Simple class to Log some messages in file
    """
    def __init__(self, logpath):
        self.logfile = open(logpath, 'w')
        self.timestamp = False
        self.unix = False

    def settimestamp(self, check=True, unix = False):
        """
        Set pretimestamp or not
        :param check: Boolean parameter
        :param unix: Boolean parameter
        """
        self.timestamp = check
        self.unix = unix

    def log(self, message):
        """
        Main method
        :param message: Type 'str'
        """
        if self.timestamp:
            if self.unix:
                message = str(time.time()) + message
            else:
                message = time.asctime() + message
        self.logfile.write(message)

    def teardown(self):
        """
        Close the writer
        """
        self.logfile.close()

    def __del__(self):
        self.logfile.close()


class Tokenizer:
    """
    Class take a string imagine of text and tokenize it.
    This instance just crop text to different words
    Class have a simple tokenize method and need some extensions.
    Also class include some simplifier.
    """

    text = None
    tokens = []

    def tokenize(self, text):
        self.text = text
        self._simplify()
        self.tokens = self.text.split(' ')
        return self.tokens

    def _regsimplify(self):
        symbols = re.compile(r'\W+')
        self.text = symbols.sub(' ', self.text)

    def _simplify(self):
        self.text = self.text.lower()
        self.text = Cucco().normalize(self.text)


class ChunkTokenizer(Tokenizer):
    """
    Tokenize text to chunks with length 'length' and interferention 'inter'
    """

    def __init__(self, length=50, inter=25):
        self.length = length
        self.inter = inter

    def tokenize(self, text):
        self.text = text
        self._regsimplify()
        words = self.text.split(' ')
        for i in range(self.length, len(words), self.length-self.inter):
            token = self._removetwin(words[i-self.length:i])
            self.tokens.append(token)
        return self.tokens

    def _removetwin(self, token):
        res = []
        for i in token:
            if i not in res:
                res.append(i)
        return res


class QueryGenerator:
    """
    Class generate query to search engine from list-token
    To generating using Word2Vec model
    """
    model = None
    length = 4

    def __init__(self, model):
        self.model = model

    def getquery(self, token, length=length):
        paars = []
        triplets = []
        for word in token:
            try:
                vec = self.model[word]
                paars.append((word, vec))
            except KeyError:
                continue
        mean = np.mean([b for a, b in paars], axis=0)
        for item in paars:
            word, vec = item
            dist = np.linalg.norm(vec-mean)
            triplets.append((word, vec, dist))
        triplets.sort(key=self._getthird)
        query = [a for a, b, c in triplets[-min(length, len(triplets)):]]
        return ' '.join(query)

    def _getthird(self, tpl):
        return tpl[2]


class SourceRetrieval:
    CHATNOIR = 'http://webis15.medien.uni-weimar.de/proxy/chatnoir/batchquery.json'
    CLUEWEB = 'http://webis15.medien.uni-weimar.de/proxy/clueweb/id/'
    text = None
    nor = None
    filelog = None
    tokens = None
    querygen = None
    dtoken = None

    suspdoc_id = None

    downloads = 0
    successes = 0
    usedurl = []

    def __init__(self, suspdoc, num_of_results, dtoken, success_log, file_log, tokenizer, querygen):
        try:
            self.suspdoc = suspdoc
            file = codecs.open(suspdoc, 'r', 'utf-8')
            self.text = file.read()
            file.close()
        except FileNotFoundError:
            print('File not found!')
            exit()

        self.nor = num_of_results
        self.dtoken = dtoken
        self.slog = success_log
        self.filelog = file_log

        self.tokens = tokenizer.tokenize(self.text)

        self.querygen = querygen

    def retrieve(self):
        print(time.asctime() + 'Running process for doc ' + self.suspdoc)
        self.suspdoc_id = int(self.suspdoc[-7:-4])  # Extracts the three digit ID.
        for token in self.tokens:
            #Get search-query from token
            query = self.querygen.getquery(token)

            #Pose query to search-engine and get results
            results = self.pose_query(query)

            #Log the query event
            self.filelog.log(query)

            if results["chatnoir-batch-results"][0]["results"] == 0:
                continue  # The query returned no results.
            reslist = self.download_results(results)
            for i in reslist:
                download, download_url = i
                if download_url not in self.usedurl:
                    self.usedurl.append(download_url)
                    # Log the download event.
                    self.filelog.log(download_url)
                    self.downloads += 1
                    # Check the text alignment oracle's verdict about the download.
                    print(download_url)
                    self.check_oracle(download, download_url, query)
        info = "In document found " + str(self.successes) + " sucesses, " + str(self.downloads)
        self.slog.log(info)
        print(time. asctime() + 'End process for doc ' + self.suspdoc)

    def pose_query(self, query):
        print(time.asctime() + "Posing query: " + query)
        json_query = u"""
                {{
                   "max-results": {num_of_res},
                   "suspicious-docid": {suspdoc_id},
                   "queries": [
                     {{
                       "query-string": "{query}"
                     }}
                   ]
                }}
                """.format(suspdoc_id=self.suspdoc_id, query=query, num_of_res=self.nor)
        json_query = \
            unicodedata.normalize("NFKD", json_query).encode("ascii", "ignore")
        try:
            request = urllib2.Request(self.CHATNOIR, json_query)
            request.add_header("Content-Type", "application/json")
            request.add_header("Accept", "application/json")
            request.add_header("Authorization", dtoken)
            request.get_method = lambda: 'POST'
            response = urllib2.urlopen(request)
            results = simplejson.loads(response.read())
            response.close()
            return results
        except urllib2.HTTPError:
            print('Found HTTP-error!')

        print(time.asctime() + "Query posed!")

    def download_results(self, results):
        print(time.asctime() + "Download results")
        """ Downloads [num of res] first-ranked results from a given result list. """
        reslist = []
        for first_result in results["chatnoir-batch-results"][0]["result-data"]:
            document_id = first_result["longid"]
            document_url = first_result["url"]
            request = urllib2.Request(self.CLUEWEB + str(document_id))
            request.add_header("Accept", "application/json")
            request.add_header("Authorization", dtoken)
            request.add_header("suspicious-docid", str(self.suspdoc_id))
            request.get_method = lambda: 'GET'

            try:
                response = urllib2.urlopen(request)
                download = simplejson.loads(response.read())
                response.close()
                reslist.append((download, document_url))
            except urllib2.HTTPError:
                print('Found HTTP-error!')
        print(time.asctime() + "Results downloaded")
        return reslist

    def check_oracle(self, download, download_url, query):
        """ Checks is a given download is a true positive source document,
            based on the oracle's decision. """
        if download["oracle"] == "source":
            print(time.asctime() + " Success: a source has been retrieved.")
            self.slog.log("Success: " + download_url + "\n Query:" + query + "\n")
            self.successes += 1
        else:
            print(time.asctime() + " Failure: no source has been retrieved.")






def startretrieval(suspdir, outdir, dtoken, model, settings=(6, 50, 25, 3)):
    outdir2 = outdir
    try:
        outdir2 = outdir + "set_" + str(settings[0]) + "_" + str(settings[1]) + \
                 "_" + str(settings[2]) + "_" + str(settings[3]) + "/"
        os.makedirs(outdir2)
    except FileExistsError:
        print('Directory is already exist')

    slog = Logger(outdir2+'success.log')
    slog.settimestamp()

    querygen = QueryGenerator(model)
    tokenizer = ChunkTokenizer(settings[1], settings[2])

    print(time.asctime() + u""" Example with settings: 
                    Chunk: {chunk} 
                    Intersect: {inter}
                    Words in query: {wordNum}
                    Num of queries: {query} 
                    started.
                  """.format(chunk=settings[1], inter=settings[2], wordNum=settings[0], query=settings[3]))

    suspdocs = glob.glob(suspdir + os.sep + 'suspicious-document???.txt')
    for suspdoc in suspdocs:
        flogname = ''.join([suspdoc[:-4], '.log'])
        flogname = ''.join([outdir, os.sep, flogname[-26:]])
        filelog = Logger(flogname)
        filelog.settimestamp(unix=True)

        retrieval = SourceRetrieval(suspdoc=suspdoc, num_of_results=settings[3], dtoken=dtoken,
                                    success_log=slog, file_log=filelog, tokenizer=tokenizer, querygen=querygen)
        retrieval.retrieve()
        filelog.teardown()
    slog.teardown()


if __name__ == '__main__':
    """units = [
        (4, 50, 25, 3),
        (5, 50, 25, 3),
        (6, 50, 25, 3),
        (7, 50, 25, 3),
        (6, 70, 35, 3),
        (6, 40, 20, 3),
        (6, 60, 30, 3),
        (6, 50, 20, 3),
        (6, 50, 30, 3),
        (6, 50, 20, 4),
        (6, 50, 20, 2)
    ]"""
    units = [(5, 80, 45, 2)]

    if len(sys.argv) == 4:
        suspdir = sys.argv[1]
        outdir = sys.argv[2]
        dtoken = sys.argv[3]

        print(time.asctime() + ' Loading w2v model')
        model = kv.load_word2vec_format('../source/GoogleNews-vectors-negative300.bin', binary=True, encoding='utf-8')
        model.init_sims(replace=True)
        print(time.asctime() + ' Model loaded')

        for unit in units:
            startretrieval(suspdir, outdir, dtoken, model, unit)

    else:
        print('\n'.join(["Unexpected number of command line arguments.",
                         "Usage: ./pan13_source_retrieval_example.py {susp-dir} {out-dir} {token}"]))
