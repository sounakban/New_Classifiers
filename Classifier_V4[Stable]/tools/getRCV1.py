import unicodedata


class getRCV1:

    def __init__(self, pathV1, pathV2, testset = 1):
        if (testset > 0 and testset < 5):
            #Load RCV1V2 Paths
            self.trainfileV2 = "{}{}".format(pathV2, "TrainingData/lyrl2004_tokens_train.dat")
            open(self.trainfileV2, 'r')
            self.testfileV2 = "{}{}{}{}".format(pathV2, "TestData/lyrl2004_tokens_test_pt", testset-1, ".dat")
            open(self.testfileV2, 'r')
            #Load RCV1 Paths
            self.trainfileV1 = "{}{}".format(pathV1, "Train")
            self.testfileV1 = "{}{}{}".format(pathV1, "Test/Set", testset)
            self.readFiles()
        elif (testset == 0):
            self.path = path
            self.readAll()
        else:
            raise ValueError('Valid values of set are 1,2,3,4. Enter Valid value.')


    def readFiles(self, ):

        self.docids = []
        self.data = {}
        import os
        import xml.etree.ElementTree as ET


        for line in open(self.trainfileV2, 'r'):
            if line.startswith(".I"):
                #line = line.rstrip()
                self.docids.append(int(line.split()[1]))
            else:
                continue
        #self.docids = self.docids[:500]

        for path, subdirs, files in os.walk(self.trainfileV1):
            for name in files:
                if len(name[:-10]) > 3:
                    if int(name[:-10]) in set(self.docids):
                        currDoc = ""
                        tree = ET.parse(os.path.join(path, name))
                        file_text = tree.getroot().find('text')
                        for para in file_text:
                            currDoc = " ".join( (currDoc, para.text.encode('ascii','ignore')) )
                        self.data[int(name[:-10])] = currDoc

        self.trainDocCount = len(self.data)
        print "Reading Training files complete"

        testdocids = []
        for line in open(self.testfileV2, 'r'):
            if line.startswith(".I"):
                #line = line.rstrip()
                testdocids.append(int(line.split()[1]))
            else:
                continue
        #testdocids = testdocids[:5000]
        self.docids.extend(testdocids)

        for path, subdirs, files in os.walk(self.testfileV1):
            for name in files:
                if len(name[:-10]) > 3:
                    if int(name[:-10]) in set(testdocids):
                        currDoc = ""
                        tree = ET.parse(os.path.join(path, name))
                        file_text = tree.getroot().find('text')
                        for para in file_text:
                            currDoc = " ".join( (currDoc, para.text.encode('ascii','ignore')) )
                        self.data[int(name[:-10])] = currDoc

        self.data = [self.data[did] for did in self.docids]
        self.totalDocCount = len(self.data)
        print "Reading Testing files complete"


    def getData(self):
        return self.data

    def getDocIDs(self):
        return self.docids

    def getTrainDocCount(self):
        return self.trainDocCount

    def getTotalDocCount(self):
        return self.totalDocCount






"""
    def readAll(self):
        self.data = []
        self.docids = []

        self.trainfile = "{}{}".format(self.path, "TrainingData/lyrl2004_tokens_train.dat")
        open(self.trainfile, 'r')
        for testset in range(4):
            self.testfile = "{}{}{}{}".format(self.path, "TestData/lyrl2004_tokens_test_pt", testset, ".dat")
            open(self.testfile, 'r')

        currDoc = ""
        for line in open(self.trainfile, 'r'):
            item = line.rstrip()
            if item.startswith(".W") or len(item)==0:
                continue
            elif item.startswith(".I"):
                if len(currDoc)!=0:
                    self.data.append(currDoc)
                    self.docids.append(currID)
                    currDoc = ""
                currID = int(item.split()[1])
            else:
                currDoc = "{}{}. ".format(currDoc, item)
        self.data.append(currDoc)
        self.docids.append(currID)

        for testset in range(4):
            self.testfile = "{}{}{}{}".format(self.path, "TestData/lyrl2004_tokens_test_pt", testset, ".dat")
            currDoc = ""
            for line in open(self.testfile, 'r'):
                item = line.rstrip()
                if item.startswith(".W") or len(item)==0:
                    continue
                elif item.startswith(".I"):
                    if len(currDoc)!=0:
                        self.data.append(currDoc)
                        self.docids.append(currID)
                        currDoc = ""
                    currID = int(item.split()[1])
                else:
                    currDoc = "{}{}. ".format(currDoc, item)
            self.data.append(currDoc)
            self.docids.append(currID)


        self.trainDocCount = 160000
        self.totalDocCount = len(self.data)
"""
