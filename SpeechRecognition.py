import numpy as np
from LSTM_Layer import LstmWeight, LSTMLayer
from SoftMax_Layer import Weight, SoftMaxLayer
from CTC_Layer import CTC

from read_data import get_data

def getDictionary(p, w):
    keys=[]
    for p1 in p:
        for p2 in p1:
            if p2 not in keys:
                keys.append(p2)
    value=[i for i in range(len(keys))]
    pdict = {key: value for key , value in (zip(keys,value))}
    words=[]
    for w1 in w:
        for w2 in w1:
            if w2 not in words:
                words.append(w2)
    value=[i for i in range(len(words))]
    wdict = {key: value for key , value in (zip(words, value))}
    return pdict, wdict
def proRawData(values, di):
    newdata=[]
    for value in values:
        tmp = []
        for feature in value:
            tmp.append(di[feature])
        newdata.append(tmp)
    return newdata

def main():
    cross_path = 'data/timit/timit/cross/'
    test_path = 'data/timit/timit/test/'
    train_path = 'data/timit/timit/train/'
    #pre word
    data = get_data(cross_path)
    feature = data['features']
    phoneme = data['phonemes']
    word = data['words']

    testdata = get_data(cross_path)
    testfeature = data['features']
    testphone = data['phonemes']
    testword = data['words']

    pdict, wdict=getDictionary(phoneme, word)
    phoneme = proRawData(phoneme, pdict)
    word = proRawData(word, wdict)

    testphone = proRawData(testphone, pdict)
    testword = proRawData(testword, wdict)

    #feature->phoneme
    np.random.seed(0)
    memcell = 100
    dimx = 12
    dimy = 61

    lstmweight = LstmWeight(memcell, dimx)
    softmaxweight = Weight(memcell, dimy)
    lstmnetwork = LSTMLayer(lstmweight)
    softmaxnetwork = SoftMaxLayer(softmaxweight)
    ####################Training#######################
    epoch = 0
    maxloss = 0
    for count in range(200):
        loss = 0
        for item in range(len(feature)):
            for content in feature[item]:
                lstmnetwork.xlistAdd(content)
            output = lstmnetwork.getHmatrix()
            softmaxnetwork.outputAdd(output.T)
            ymatrix = softmaxnetwork.getYmatrix()
            ctclayer = CTC(ymatrix.T, phoneme[item])
            do,tmp = ctclayer.returndY()
            loss += tmp
            softmaxnetwork.ylist(do)
            hmatrix = softmaxnetwork.getdHmatrix()
            lstmnetwork.ylist(hmatrix.T)
            lstmnetwork.xlistRefresh()
            softmaxnetwork.outputRefresh()
            softmaxweight.changeWeight(0.001)
            lstmweight.changeWeight(0.001)
        print(loss)
        if loss>maxloss:
            maxloss=loss
    ####################Testing#######################
    for item in range(len(testfeature)):
        for content in testfeature[item]:
            lstmnetwork.xlistAdd(content)
        print('label',testphone[item])
        output = lstmnetwork.getHmatrix()
        lstmnetwork.xlistRefresh()
        print('predict',softmaxnetwork.predict(output.T))

main()

