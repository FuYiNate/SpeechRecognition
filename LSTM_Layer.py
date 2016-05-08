import random
import numpy as np
import math

def sigmoid(x):
    for i in range(len(x)):
        if x[i] < -700:
            x[i] = -700
    return 1./(1+np.exp(-x))

def assignValue(a,b,*args):
    #initialize the weight
    np.random.seed(0)
    return np.random.rand(*args)*(b-a)+a

class LstmWeight:

    def __init__(self,  memcell, dimx):
        #initialize the LSTM weight
        self.memcell = memcell
        self.dimx = dimx
        #hidden layer+input X
        mx_len = memcell+dimx

        self.W_i = assignValue(-1,1,memcell, mx_len)
        self.W_f = assignValue(-1, 1, memcell, mx_len)
        self.W_o =  assignValue(-1, 1, memcell, mx_len)
        self.W_c =  assignValue(-1, 1, memcell, mx_len)
        #bias
        self.b_i = assignValue(-1, 1, memcell)
        self.b_f = assignValue(-1, 1, memcell)
        self.b_o = assignValue(-1, 1, memcell)
        self.b_c = assignValue(-1, 1, memcell)
        #initialize derivative
        self.refreshDerivative()

    def refreshDerivative(self):
        self.dW_i = np.zeros_like(self.W_i)
        self.dW_f = np.zeros_like(self.W_f)
        self.dW_o = np.zeros_like(self.W_o)
        self.dW_c = np.zeros_like(self.W_c)
        self.db_i = np.zeros_like(self.b_i)
        self.db_f = np.zeros_like(self.b_f)
        self.db_o = np.zeros_like(self.b_o)
        self.db_c = np.zeros_like(self.b_c)

    def changeWeight(self, step=0.01):
        self.W_i -= step*self.dW_i
        self.W_f -= step*self.dW_f
        self.W_o -= step*self.dW_o
        self.W_c -= step*self.dW_c
        self.b_i -= step*self.db_i
        self.b_f-= step*self.db_f
        self.b_o -= step*self.db_o
        self.b_c -= step*self.db_c
        self.refreshDerivative()

    def refreshWeight(self):
        self.W_i*=10
        self.W_f*=10
        self.W_o*=10
        self.W_c*=10
        self.b_i*=10
        self.b_f*=10
        self.b_o*=10
        self.b_c*=10

class MemoryState:

    def __init__(self, memcell, dimx):
        #initialize i, f, o, c, s, h
        self.i = np.zeros(memcell)
        self.f = np.zeros(memcell)
        self.o = np.zeros(memcell)
        self.c = np.zeros(memcell)
        self.s= np.zeros(memcell)
        self.h = np.zeros(memcell)

        self.ds = np.zeros_like(self.s)
        self.dh = np.zeros_like(self.h)
class MemoryNode:

    def __init__(self, weight, state):
        #every memory cell has it's own weight and state, althought they share the same weight
        self.state = state
        self.weight = weight
        self.x = None
        self.x_h = None

    def ForwardStep(self, x, forget, prev_s= None, prev_h=None):
        if prev_s == None: prev_s = np.zeros_like(self.state.s)
        if prev_h == None: prev_h = np.zeros_like(self.state.h)
        self.prev_s = prev_s
        self.prev_h = prev_h
        x_h = np.hstack((x,prev_h))
        #input gate
        self.state.i = sigmoid(np.dot(self.weight.W_i,x_h)+self.weight.b_i)
        #forget gate. Control how many previous state the node should remember
        if forget:
            self.state.f = np.ones_like(self.state.i)-self.state.i
        else:
            self.state.f = sigmoid(np.dot(self.weight.W_f,x_h)+self.weight.b_f)
        #output gate
        self.state.o = sigmoid(np.dot(self.weight.W_o,x_h)+self.weight.b_o)
        #cell candidate value
        self.state.c = np.tanh(np.dot(self.weight.W_c, x_h)+self.weight.b_c)
        #cell value
        self.state.s = self.state.c*self.state.i+prev_s*self.state.f
        #hidden value
        self.state.h = np.tanh(self.state.s)*self.state.o

        self.x = x
        self.x_h = x_h

    def BackwardStep(self, dh, ds, forget):
        #calculate drivative of all the value
        ds = self.state.o*dh+ds
        dc = self.state.i*ds
        do = self.state.s*dh
        df = self.prev_s*ds
        if forget:
            di=(self.state.c-self.prev_s)*ds
        else:
            di = self.state.c*ds

        dc_input = (1. - self.state.c ** 2) * dc
        do_input = (1. - self.state.o) * self.state.o * do
        di_input = (1. - self.state.i) * self.state.i * di
        if forget:
            df_input = -di_input
        else:
            df_input = (1. - self.state.i) * self.state.f * df

        dx_h = np.zeros_like(self.x_h)

        self.weight.dW_c += np.outer(dc_input, self.x_h)
        self.weight.dW_o += np.outer(do_input, self.x_h)
        self.weight.dW_f += np.outer(df_input, self.x_h)
        self.weight.dW_i += np.outer(di_input, self.x_h)

        self.weight.db_c += dc_input
        self.weight.db_o += do_input
        self.weight.db_f += df_input
        self.weight.db_i += di_input

        dx_h += np.dot(self.weight.W_i.T, di_input)
        dx_h += np.dot(self.weight.W_f.T, df_input)
        dx_h += np.dot(self.weight.W_o.T, do_input)
        dx_h += np.dot(self.weight.W_c.T, dc_input)

        self.state.ds = ds*self.state.f
        self.dh = dx_h[self.weight.dimx:]

class LSTMLayer:

    def __init__(self, weight, timestamp=4):
        #inital network
        self.weight = weight
        self.nodelist = []
        self.xlist = []
        self.timestamp = timestamp

    def xlistAdd(self, x):
        forget = False
        self.xlist.append(x)
        state = MemoryState(self.weight.memcell, self.weight.dimx)
        self.nodelist.append(MemoryNode(self.weight, state))
        count = len(self.xlist)-1
        if count!=0 and count%self.timestamp==0:
            forget = True
        else:
            forget = False
        if count==0:
            self.nodelist[count].ForwardStep(x, forget)
        else:
            pre_s = self.nodelist[count-1].state.s
            pre_h = self.nodelist[count-1].state.h
            self.nodelist[count].ForwardStep(x, forget, pre_s, pre_h)

    def xlistRefresh(self):
        self.xlist = []
        self.nodelist = []

    def ylist(self, dhmatrix):
        count = dhmatrix.shape[0]-1

        dh = dhmatrix[count]
        ds = np.zeros(self.weight.memcell)

        if count%self.timestamp==0:
            forget = True
        else:
            forget = False

        self.nodelist[count].BackwardStep(dh, ds, forget)

        count -= 1
        while count>=0:
            if count%self.timestamp==0:
                forget = True
            else:
                forget = False

            dh = dhmatrix[count]
            dh += self.nodelist[count+1].state.dh
            ds = self.nodelist[count+1].state.ds
            self.nodelist[count].BackwardStep(dh,ds, forget)
            count -= 1

    def getHmatrix(self):
        hmatrix = np.column_stack(i.state.h for i in self.nodelist)
        return hmatrix
