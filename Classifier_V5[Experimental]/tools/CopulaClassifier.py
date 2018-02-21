import numpy
from multiprocessing import Process, Queue, cpu_count
thread_count = int(cpu_count()*0.75)

from copula_utils import classify

class CopulaClassifier:

    def __init__(self, corcoeff, vocab, priors = []):
        if len(priors) != 0:
            if len(corcoeff) != len(vocab) or len(vocab) != len(priors):
                print "Check input values of CopulaClassifier initialization"
        else:
            if len(corcoeff) != len(vocab):
                print "Check input values of CopulaClassifier initialization"
        self.corcoeff = corcoeff
        self.vocab = vocab
        self.priors = priors



    def predict_multiclass(self, test_docs):
        from copula_utils import pred_maxScore
        scores_dict = {}
        scores_list = []
        predictions_list = []
        que = Queue()

        if thread_count < len(test_docs):
            div = (len(test_docs)/thread_count)+1
            processes = []
            for i in range(thread_count):
                end = (i+1)*div
                if end > len(test_docs):
                    end = len(test_docs)
                processes.append(Process(target=classify, args=( self.corcoeff, self.vocab, self.priors, test_docs[i*div:end], i, que)))
            for pro in processes:
                pro.start()
            for pro in processes:
                temp = que.get()
                scores_dict[temp[0]] = temp[1:]
            for pro in processes:
                pro.join()
            for i in range(len(processes)):
                scores_list.extend(scores_dict[i])
        else:
            classify(self.corcoeff, self.vocab, self.priors, test_docs, 0, que)
            scores_list.extend(que.get()[1:])
        self.scores_list = scores_list

        predictions_list = map(pred_maxScore, scores_list)
        return numpy.array(predictions_list)





    def predict_multilabel(self, test_docs):
        from Thresholding import M_Cut, M_Cut_mod, M_Cut_mod2
        scores_dict = {}
        scores_list = []
        que = Queue()

        if thread_count < len(test_docs):
            div = (len(test_docs)/thread_count)+1
            processes = []
            for i in range(thread_count):
                end = (i+1)*div
                if end > len(test_docs):
                    end = len(test_docs)
                processes.append(Process(target=classify, args=( self.corcoeff, self.vocab, self.priors, test_docs[i*div:end], i, que)))
            for pro in processes:
                pro.start()
            for pro in processes:
                temp = que.get()
                scores_dict[temp[0]] = temp[1:]
            for pro in processes:
                pro.join()
            for i in range(len(processes)):
                scores_list.extend(scores_dict[i])
        else:
            classify(self.corcoeff, self.vocab, self.priors, test_docs, 0, que)
            scores_list.extend(que.get()[1:])
        self.scores_list = scores_list

        predictions_list = map(M_Cut, scores_list)
        #predictions_list = map(M_Cut_mod, scores_list)
        #predictions_list = map(M_Cut_mod2, scores_list)
        return numpy.array(predictions_list)





    def predict_multilabelBR(self, test_docs, all_terms = []):
        from copula_utils import pred_ScoreBR
        scores_dict = {}
        scores_list = []
        que = Queue()

        if thread_count < len(test_docs):
            div = (len(test_docs)/thread_count)+1
            processes = []
            for i in range(thread_count):
                end = (i+1)*div
                if end > len(test_docs):
                    end = len(test_docs)
                processes.append(Process(target=classify, args=( self.corcoeff, self.vocab, self.priors, test_docs[i*div:end], i, que, all_terms)))
            for pro in processes:
                pro.start()
            for pro in processes:
                temp = que.get()
                scores_dict[temp[0]] = temp[1:]
            for pro in processes:
                pro.join()
            for i in range(len(processes)):
                scores_list.extend(scores_dict[i])
        else:
            classify(self.corcoeff, self.vocab, self.priors, test_docs, 0, que)
            scores_list.extend(que.get()[1:])
        self.scores_list = scores_list

        predictions_list = map(pred_ScoreBR, scores_list)
        return numpy.array(predictions_list)




    def get_scores():
        return self.scores_list
