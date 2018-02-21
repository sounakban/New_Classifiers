class getRCV1V2:

    def __init__(self, path, testset = 1):
        if (testset > 0 and testset < 5):
            self.trainfile = "{}{}".format(path, "TrainingData/lyrl2004_tokens_train.dat")
            open(self.trainfile, 'r')
            self.testfile = "{}{}{}{}".format(path, "TestData/lyrl2004_tokens_test_pt", testset-1, ".dat")
            open(self.testfile, 'r')
            self.readFiles()
        elif (testset == 0):
            self.path = path
            self.readAll()
        else:
            raise ValueError('Valid values of set are 1,2,3,4. Enter Valid value.')

    def readFiles(self, ):
        self.data = []
        self.docids = []

        currDoc = ""
        count = 0
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
                """
                if count >= 500:
                    break
                count = count + 1
                #"""
            else:
                currDoc = "{}{}. ".format(currDoc, item)
        self.data.append(currDoc)
        self.docids.append(currID)
        self.trainDocCount = len(self.data)

        currDoc = ""
        count = 0
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
                """
                if count >= 500:
                    break
                count = count + 1
                #"""
            else:
                currDoc = "{}{}. ".format(currDoc, item)
        self.data.append(currDoc)
        self.docids.append(currID)
        self.totalDocCount = len(self.data)


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



    def getData(self):
        return self.data

    def getDocIDs(self):
        return self.docids

    def getTrainDocCount(self):
        return self.trainDocCount

    def getTotalDocCount(self):
        return self.totalDocCount
