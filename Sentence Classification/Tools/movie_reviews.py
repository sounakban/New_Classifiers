def get_PosReviews():
    fl_path = "/home/sounak/Resources/Data/rt-polaritydata/rt-polarity.pos"
    lines = open(fl_path, encoding='latin1').read().encode('ascii', 'ignore').splitlines()
    lines = [line.decode('utf-8') for line in lines]    #Convert bytes to strings
    return lines


def get_NegReviews():
    fl_path = "/home/sounak/Resources/Data/rt-polaritydata/rt-polarity.neg"
    lines = open(fl_path, encoding='latin1').read().encode('ascii', 'ignore').splitlines()
    lines = [line.decode('utf-8') for line in lines]    #Convert bytes to strings
    return lines
