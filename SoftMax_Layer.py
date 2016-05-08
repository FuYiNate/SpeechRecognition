import numpy as np

def softmax(x):
    xt = np.exp(x)
    return xt/np.sum(xt)

class Weight:

    def __init__(self, memcell, dimy):
        self.memcell = memcell
        self.dimy = dimy
        self.V = np.random.rand(dimy, memcell)*(2)-1
        self.dV = np.zeros_like(self.V)

    def changeWeight(self, step=0.01):
        self.V -= step*self.dV
        self.dV = np.zeros_like(self.V)

class SoftMaxState:
    def __init__(self, memcell, dimy):
        self.o = np.zeros(dimy)
        self.dh = np.zeros(memcell)
        self.h = np.zeros(memcell)
class SoftMaxNode:

    def __init__(self, weight, state):
        self.state = state
        self.weight = weight
    def ForwardStep(self, h):
        self.state.o = softmax(np.dot(self.weight.V,h))
        self.state.h = h
    def BackwardStep(self, dy):
        dv = np.outer(dy, self.state.h)
        self.weight.dV += dv
        self.state.dh = np.dot(dy, self.weight.V)

class SoftMaxLayer:

    def __init__(self, weight):
        self.weight = weight
        self.nodelist = []
        self.outputlist = []

    def outputAdd(self, output):
        self.outputlist = output
        for i in range(output.shape[0]):
            state = SoftMaxState(self.weight.memcell,self.weight.dimy)
            self.nodelist.append(SoftMaxNode(self.weight, state))
            self.nodelist[i].ForwardStep(output[i])

    def predict(self, output):
        self.outputlist = output
        for i in range(output.shape[0]):
            state = SoftMaxState(self.weight.memcell,self.weight.dimy)
            self.nodelist.append(SoftMaxNode(self.weight, state))
            self.nodelist[i].ForwardStep(output[i])
        y = np.column_stack(i.state.o for i in self.nodelist)
        tmp=[]
        for array in y.T:
            tmp.append(np.argmax(array))
        predict =[]
        now = 0
        for element in tmp:
            if element!=now:
                predict.append(element)
                now = element
        result = []
        for element in tmp:
            if element!=0:
                result.append(element)
        return result

    def outputRefresh(self):
        self.nodelist = []
        self.outputlist = []

    def ylist(self, dymatrix):
        for i in range(dymatrix.shape[0])[::-1]:
            self.nodelist[i].BackwardStep(dymatrix[i])

    def getYmatrix(self):
        ymatrix = np.column_stack(i.state.o for i in self.nodelist)
        return ymatrix

    def getdHmatrix(self):
        dhmatrix = np.column_stack(i.state.dh for i in self.nodelist)
        return dhmatrix
