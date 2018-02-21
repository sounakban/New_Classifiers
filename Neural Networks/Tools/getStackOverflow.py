class getStackOverflow:

    def __init__(self, path):
        self.datafile = "{}{}".format(path, "rawText/title_StackOverflow.txt")
        open(self.datafile, 'r')
        self.targetfile = "{}{}".format(path, "rawText/label_StackOverflow.txt")
        open(self.targetfile, 'r')
        self.readFiles()

    def readFiles(self, ):
        self.data = []
        self.label = []

        for line in open(self.datafile, 'r'):
            item = line.rstrip()
            self.data.append(item)

        for line in open(self.targetfile, 'r'):
            #Convert to tuple
            #item = ( int(line.rstrip()), )
            item = int(line.rstrip())
            self.label.append(item)


    def getData(self):
        return self.data

    def getTarget(self):
        return self.label

    def getTotalDocCount(self):
        return self.totalDocCount
