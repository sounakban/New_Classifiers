def M_Cut(scores):
    #implement M-Cut thresholding
    sorted_scores = list(scores)
    sorted_scores.sort()
    max_diff = scr1 = scr2 = 0
    for i in range(len(sorted_scores)-1):
        if max_diff < abs(sorted_scores[i]-sorted_scores[i+1]):
            scr1 = sorted_scores[i]
            scr2 = sorted_scores[i+1]
            max_diff = abs(sorted_scores[i]-sorted_scores[i+1])
    thresh = (scr1+scr2)/2
    #print "Threshold: ", thresh, scr1, scr2
    pred = [0]*len(scores)
    classes = [i for i, x in enumerate(scores) if x > thresh]
    for i in classes:
        pred[i] = 1
    return pred


def M_Cut_mod(scores):
    #implement Mod on M-Cut thresholding
    sorted_scores = list(scores)
    sorted_scores.sort(reverse=True)
    max_diff = scr1 = scr2 = 0
    pos = 0
    for i in range(len(sorted_scores)-1):
        if max_diff < abs(sorted_scores[i]-sorted_scores[i+1]):
            scr1 = sorted_scores[i]
            scr2 = sorted_scores[i+1]
            max_diff = abs(sorted_scores[i]-sorted_scores[i+1])
            pos = i
    avg = abs(sorted_scores[0]-sorted_scores[len(sorted_scores)-1])/float(len(sorted_scores))
    if pos==0:
        avg += float(max_diff-avg)
        step = 0
    else:
        step = float(max_diff-avg)/pos
    """
    diff = []
    for i in range(len(sorted_scores)-1):
        diff.append(abs(sorted_scores[i]-sorted_scores[i+1]))
    sorted_diff = list(diff)
    sorted_diff.sort(reverse=True)
    tmp = sel_diff = 0
    for i in range(len(sorted_diff)-1):
        if temp < abs(sorted_diff[i]-sorted_diff[i+1]):
            sel_diff = sorted_diff[i]
            diff2 = sorted_diff[i+1]
            temp = abs(sorted_diff[i]-sorted_diff[i+1])
    """
    thresh = (scr1+scr2)/2
    for i in range(pos+1):
        if float(abs(sorted_scores[i] - sorted_scores[i+1])) > (avg + (step * i) + 1):
            thresh = (sorted_scores[i] + sorted_scores[i+1])/2
    #print "Threshold: ", thresh, scr1, scr2
    pred = [0]*len(scores)
    classes = [i for i, x in enumerate(scores) if x > thresh]
    for i in classes:
        pred[i] = 1
    return pred



def M_Cut_mod2(scores):
    #implement Mod on M-Cut thresholding
    sorted_scores = list(scores)
    sorted_scores.sort(reverse=True)
    max_diff = scr1 = scr2 = 0
    pos = 0
    for i in range(len(sorted_scores)-1):
        if max_diff < abs(sorted_scores[i]-sorted_scores[i+1]):
            scr1 = sorted_scores[i]
            scr2 = sorted_scores[i+1]
            max_diff = abs(sorted_scores[i]-sorted_scores[i+1])
            pos = i
    thresh = (scr1+scr2)/2

    slope = (scr1 - sorted_scores[0])/pos
    c = sorted_scores[0]
    for i in reversed(range(pos)):
        if sorted_scores[i] > ((slope * i) + c):
            thresh = (sorted_scores[i] + sorted_scores[i+1])/2

    #print "Threshold: ", thresh, scr1, scr2
    pred = [0]*len(scores)
    classes = [i for i, x in enumerate(scores) if x > thresh]
    for i in classes:
        pred[i] = 1
    return pred


"""
def P_Cut(all_scores):
    for scores in all_scores:
"""
