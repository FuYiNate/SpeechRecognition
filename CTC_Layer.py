import numpy as np

class CTC:
    def __init__(self, y, l):
        _l = [0]
        for i in l:
            if i!=0:
                _l.append(i)
                _l.append(0)
        self.y = y
        self.l = _l
        self.T = len(y)
        self.alpha = self.ForwardAlpha()
        self.beta = self.Backwardbeta()
    def ForwardAlpha(self):
        alpha = np.zeros((self.T+2,len(self.l)+2))
        alpha[1,1]=self.y[0][self.l[0]]
        alpha[1,2]=self.y[0][self.l[1]]
        C_t = alpha[1,1]+alpha[1,2]
        alpha[1,1]/=C_t
        alpha[1,2]/=C_t
        for t in range(1,self.T):
            a_t = t+1
            C_t=0
            for s in range(len(self.l)):
                a_s = s+1
                if  self.l[s]==0 or (s-2>=0 and self.l[s]==self.l[s-2]):
                    alpha[a_t,a_s]=(alpha[a_t-1][a_s]+alpha[a_t-1][a_s-1])*self.y[t][self.l[s]]
                else:
                    alpha[a_t,a_s]=(alpha[a_t-1][a_s]+alpha[a_t-1][a_s-1]+alpha[a_t-1][a_s-2])*self.y[t][self.l[s]]
                C_t+=alpha[a_t][a_s]
            for s in range(len(self.l)):
                a_s=s+1
                alpha[a_t,a_s]/=C_t
        return alpha
    def Backwardbeta(self):
        beta = np.zeros((self.T+2,len(self.l)+2))
        beta[self.T+1][len(self.l)+1]=0
        beta[self.T][len(self.l)]=self.y[self.T-1][self.l[len(self.l)-1]]
        beta[self.T][len(self.l)-1]=self.y[self.T-1][self.l[len(self.l)-2]]
        D_t = beta[self.T][len(self.l)]+beta[self.T][len(self.l)-1]
        beta[self.T][len(self.l)]/=D_t
        beta[self.T][len(self.l)-1]/=D_t
        for t in range(self.T-1)[::-1]:
            b_t=t+1
            D_t=0
            for s in range(0,len(self.l))[::-1]:
                b_s = s+1
                if self.l[s]==0 or (s+2< len(self.l) and self.l[s]==self.l[s+2]):
                    beta[b_t][b_s]=(beta[b_t+1][b_s]+beta[b_t+1][b_s+1])*self.y[t][self.l[s]]
                else:
                    beta[b_t][b_s]=(beta[b_t+1][b_s]+beta[b_t+1][b_s+1]+beta[b_t+1][b_s+2])*self.y[t][self.l[s]]
                D_t+=beta[b_t][b_s]
            for s in range(len(self.l)):
                b_s=s+1
                beta[b_t][b_s]/=D_t
        return beta
    def getUniqueElement(self):
        klist=[]
        for x in self.l:
            if x not in klist:
                klist.append(x)
        return klist
    def getLab(self,k):
        lab=[]
        for i in range(len(self.l)):
            if(self.l[i]==k):
                lab.append(i+1)
        return lab
    def getSig(self,k,t):
        lab = self.getLab(k)
        sig=0
        for s in lab:
            sig+=self.alpha[t][s]*self.beta[t][s]
        return sig
    def returnloss(self):
        return np.ln()
    def returndY(self):
        predict=0
        dy = np.zeros_like(self.y)
        klist = self.getUniqueElement()
        for t in range(self.T):
            zt = 0
            for s in range(len(self.l)):
                zt+=self.alpha[t+1][s+1]*self.beta[t+1][s+1]/self.y[t][self.l[s]]
            if t==0:
                predict=zt
            for k in klist:
                dy[t][k]=self.y[t][k]-self.getSig(k,t+1)/(self.y[t][k]*zt)
        return dy,zt
    def loss(self):
        l = self.l
        _l=[]
        for i in l:
            if i != 0:
                _l.append(i)
        l=_l
        y = self.y
        valuematrix = np.zeros((len(l)+1,len(y)+1))
