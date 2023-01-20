"""
Created on Tue Aug 31 19:40:10 2021

@author: sasak

DVR方針
1.ho基底を用いてRを対角化する。
2.固有値のセットRiをつくる。
3.固有ベクトルを並べて行列Tをつくる。
4.ポテンシャルVの行列要素Vstに対してV(Ri)*Tsi*Titと変換する。
5.Hamiltonian行列を対角化する。

ver1.1.0からの変更点
・DVRを用いたR^-nによる減衰項を追加
・stretchに対してDVRを用いたMorseポテンシャルの実装
・Tsの行列要素の誤りを訂正
・kxxzの記入し忘れ訂正

ver3.0.0からの変更点
・行列要素項 kzz -> kzz*az**2
・計算する行列のmをcalcmを使ってスクリプトの上の方で操作できるようにした

ver3.1.0からの変更点
・\partial_R^2 V3の項を追加

ver.3.2.0からの変更点
・Hamiltonian項ごとにforネストを分解。やや速くなる。
・Vam, VamR項をrhoの2次から1次に変更(2022.07.20)

ver.3.3.0からの変更点
・行列要素の部分をclassを使って大きく書き換えた
・行列の2D配列作成を内包表記を使って高速化
・行列の配列作成は全て1つのforネストに突っ込んだ, 高速化
・対称化基底で正しい固有ベクトルを得られるようになった(eigen_vecにユニタリー行列をかけるのを忘れてた)
・対称化なし、対称化固有ベクトルあり、対称化固有ベクトルなしのパターンを選べるようにした。
・run timeを記録するようにした

j = 8ではEしぇい行列が正しくならない(行列要素が正しくない？)
-> 現時点ではj = 7までしか対称化基底での対角化を使えない。

※回す前にHamiltonianの項が全部揃ってるか確認しような
"""

import scipy.linalg
import numpy as np
from sympy.physics.wigner import wigner_3j
from sympy.physics.wigner import wigner_d_small
import time
import pprint
import wigner_d_small_pihalf
 
#global parameter
calcm = 0
diag_coeff = 0.2
#matrix size
n0 = 5
npls = 4
nmns = 4
j = 5

#potential parameter
(V3, V4, V6, V7) = (-50.113, 2.8980, 0, 0)
(VR3, VR4) = (-191.57, 11.360)
(VRR3, VRR4) = (322.19, 0)
(Vrho3, Vrho4) = (42.15, -1.0367)
Vam31 = + 477.34
VamR31 = - 2173120  #VamとVamRは相対符号があっていれば、パラメータは+, -どちらでも固有値は同じ
(kxx, kzz, kxxz, az) = (145.14, 622.44, -183.70, 1.4440)
kxxxx = 0
nn = 6
Re = 3.6316
B = 5.241       #methane monomer's rotational constant
shbaromega =  10/(2*np.pi*2.998)*np.sqrt(6.02*1.98645*2/1.328*kzz*az**2)  #Bz-CH4 #pseudo diatom molecule's reduced masss (hbar omega)/4
bhbaromega =  10/(2*np.pi*2.998)*np.sqrt(6.02*1.98645*2*10*kxx*(1/13.28 + 0.1497))  #Bz-CH4 #pseudo diatom molecule's reduced masss (hbar omega)/4
#B = 2.621 #Bz-CD4
#shbaromega = 10/(2*np.pi*2.998)*np.sqrt(6.02*1.98645*2*10/15.92*kzz)  #Bz-CD4
#bhbaromega =  10/(2*np.pi*2.998)*np.sqrt(6.02*1.98645*2*10*kxx*(1/15.92 + 0.1497))  #Bz-CH4  

print('n0 =', n0, ' npls =', npls, ' nmns =', nmns, ' j =', j, 'm = ', calcm)
print(' internal rotation parameter','(V3, V4, V6, V7) =', (V3, V4, V6, V7))
print(' R coupling parameter','(VR3, VR4) =', (VR3, VR4))
print(' rho coupling parameter','(Vrho3, Vrho4) =', (Vrho3, Vrho4))


class MatrixParameter:
    def __init__(self, m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn):   #l = m - mBで従属しているのでlは引数に必要ない。
        self.m = m
        self.jrow = jrow
        self.mBrow = mBrow
        self.krow = krow
        self.n0row = n0row
        self.vrow = vrow
        #self.lrow = lrow
        self.lrow = m - mBrow
        self.jcolumn = jcolumn
        self.mBcolumn = mBcolumn
        self.kcolumn = kcolumn
        self.n0column = n0column
        self.vcolumn = vcolumn
        #self.lcolumn = lcolumn
        self.lcolumn = m - mBcolumn
        
        self.nplsrow = (self.vrow + self.lrow)/2
        self.nmnsrow = (self.vrow - self.lrow)/2
        self.nplscolumn = (self.vcolumn + self.lcolumn)/2
        self.nmnscolumn = (self.vcolumn - self.lcolumn)/2
        
    def size(self):
        #generate matrix
        dim = [0]
        allowl = [0]
        allowmB = [0]
        for ll in range(- nmns, npls + 1):
            for mB in range(- j, j + 1):
                
                jdim = []
                for num in range(0, j + 1):
                    jd = (2 * abs(num) + 1)
                    jdim.append(jd)
                    
                mminus = []
                for mm in range(0, abs(mB) + 1):
                    if mm == 0:
                        a = 0
                    #elif mm == 1:
                    #    a = 0
                    else:
                        a = (2 * (abs(mm) - 1) + 1)
                    mminus.append(a)
                    
                if ll + mB == self.m:
                    d = (npls + 1 - abs(ll))*(sum(jdim)-sum(mminus))
                    dim.append(d)
                    allowl.append(ll)
                    allowmB.append(mB)
        dimension = sum(dim)*(n0 + 1)
        return dimension, dim, allowl, allowmB

class tools(MatrixParameter):
        
    def symbol(self, p, q):
        symbol1 = wigner_3j( self.jrow, p, self.jcolumn,
                            -self.krow, q, self.kcolumn)
        
        symbol2 = wigner_3j( self.jrow,  p, self.jcolumn,
                            -self.mBrow, 0, self.mBcolumn)
        
        symbol = symbol1*symbol2
        return symbol
    
    def symbol_couple(self, p, q, m_couple):
        symbolpls1 = wigner_3j( self.jrow, p, self.jcolumn,
                               -self.krow, q, self.kcolumn)
        
        symbolpls2 = wigner_3j(  self.jrow,        p, self.jcolumn,
                               -self.mBrow, m_couple, self.mBcolumn)
        symbolpls = symbolpls1*symbolpls2
        return symbolpls
    
def DVR():
    x = np.zeros((n0 + 1, n0 + 1))
    for i in range (1, n0 + 1):
        x[i, i - 1] = np.sqrt(i)    #creation operator
        x[i - 1, i] = np.sqrt(i)    #annihilation operator
    #print('position matrix\n{}\n'.format(x))
    x = np.sqrt(shbaromega/(4*kzz*az**2))*x
    Ri, Ri_vec = np.linalg.eigh(x)  
    print("!! DVR diagonalized !!")
    #print('Ri eigen value\n{}\n'.format(Ri))
    #print(Ri_vec.real)
    T = Ri_vec.real
    return Ri, T

Ri = DVR()[0]
Ri_vec = DVR()[1]

class MatrixElement(tools):

    def Ts(self):
        if self.jrow == self.jcolumn and self.krow == self.kcolumn and self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnscolumn: 
            if self.n0row == self.n0column:
                Ts = 2*self.n0column + 1
            elif self.n0row == self.n0column + 2:
                Ts = - np.sqrt((self.n0column + 1)*(self.n0column + 2))
            elif self.n0row == self.n0column - 2:
                Ts = - np.sqrt(self.n0column*(self.n0column - 1))
            else:
                Ts = 0
        else:
            Ts = 0
        return Ts
    
    def Tb(self):    
        if self.jrow == self.jcolumn and self.krow == self.kcolumn and self.n0row == self.n0column:
            if self.nplsrow == self.nplscolumn + 1 and self.nmnsrow == self.nmnscolumn + 1:
                Tb = - 2*np.sqrt((self.nplscolumn + 1)*(self.nmnscolumn + 1))
            elif  self.nplsrow == self.nplscolumn - 1 and self.nmnsrow == self.nmnscolumn - 1:
                Tb = - 2*np.sqrt((self.nplscolumn)*(self.nmnscolumn))
            elif  self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnscolumn:
                Tb = 2*(self.nplscolumn + self.nmnscolumn + 1)
            else:
                Tb = 0
        else:
            Tb = 0
        return Tb
    
    def Tr(self):
        if self.jrow == self.jcolumn and self.krow == self.kcolumn and self.n0row == self.n0column and self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnscolumn: 
            Tr = self.jrow*(self.jrow + 1)
        else: 
            Tr = 0
        return Tr
    
    def Vs(self):
        #unitary matrix product
        VVs = 0
        for i in range(0, n0 + 1):
            VVs += (kzz*(1 - np.exp(-az*Ri[i]))**2)*Ri_vec[self.n0row, i]*Ri_vec[self.n0column, i]
        #Vs
        if self.jrow == self.jcolumn and self.krow == self.kcolumn and self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnscolumn:
            Vs = VVs
        else:
            Vs = 0
        return Vs
    
    def Vb(self):
        if self.jrow == self.jcolumn and self.krow == self.kcolumn and self.n0row == self.n0column:
            if self.nplsrow - self.nplscolumn == + 1 and self.nmnsrow - self.nmnscolumn == + 1:
                Vb = 2*np.sqrt((self.nplscolumn + 1)*(self.nmnscolumn + 1))                  
            elif  self.nplsrow - self.nplscolumn == - 1 and self.nmnsrow - self.nmnscolumn == - 1:
                Vb = 2*np.sqrt((self.nplscolumn)*(self.nmnscolumn))
            elif  self.nplsrow - self.nplscolumn == 0 and self.nmnsrow - self.nmnscolumn == 0:
                Vb = 2*(self.nplscolumn + self.nmnscolumn + 1)
            else:
                Vb = 0          
        else:
            Vb = 0
        return Vb
    
    def Vr(self, p, q):
        if self.n0row == self.n0column and self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnscolumn: 
            Vr = ((-1)**(self.mBrow - self.kcolumn))*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.symbol(p, q)
        else:
            Vr = 0
        return Vr
    
    def Vsb(self):
        #unitary matrix product
        VVsb = 0
        for i in range(0, n0 + 1):
            VVsb += kxxz*(1 - np.exp(-az*Ri[i]))*Ri_vec[self.n0row, i]*Ri_vec[self.n0column, i]  
        #Vsb
        if self.jrow == self.jcolumn and self.krow == self.kcolumn:
            if self.nplsrow == self.nplscolumn + 1 and self.nmnsrow == self.nmnscolumn + 1:
                Vsb = 2*np.sqrt((self.nplscolumn + 1)*(self.nmnscolumn + 1))*VVsb
            elif  self.nplsrow == self.nplscolumn - 1 and self.nmnsrow == self.nmnscolumn - 1:
                Vsb = 2*np.sqrt((self.nplscolumn)*(self.nmnscolumn))*VVsb
            elif  self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnscolumn:
                Vsb = 2*(self.nplscolumn + self.nmnscolumn + 1)*VVsb
            else:
                Vsb = 0
        else:
            Vsb = 0
        return Vsb
    
    def Vsr(self, p, q):
        if self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnscolumn:
            if self.n0row == self.n0column + 1:
                Vsr = (np.sqrt(self.n0column + 1)
                       *((-1)**(self.mBcolumn - self.kcolumn))*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.symbol(p, q))
            elif  self.n0row == self.n0column - 1:
                Vsr = (np.sqrt(self.n0column)
                       *((-1)**(self.mBcolumn - self.kcolumn))*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.symbol(p, q))
            elif  self.n0row > self.n0column + 2:
                Vsr = 0
            elif  self.n0row < self.n0column - 2:
                Vsr = 0
            elif self.n0row == self.n0column:
                Vsr = 0
            else:
                Vsr = 0
        else:
            Vsr = 0
        return Vsr
        
    def Vssr(self, p, q):
        if self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnscolumn:
            if self.n0row == self.n0column:
                Vssr = ((2*self.n0column + 1)
                        *((-1)**(self.mBcolumn - self.kcolumn))*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.symbol(p, q))
            elif  self.n0row == self.n0column + 2:
                Vssr = (np.sqrt((self.n0column + 1)*(self.n0column + 2))
                        *((-1)**(self.mBcolumn - self.kcolumn))*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.symbol(p, q))
            elif  self.n0row == self.n0column - 2:
                Vssr = (np.sqrt(self.n0column*(self.n0column - 1))
                        *((-1)**(self.mBcolumn - self.kcolumn))*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.symbol(p, q))
            else:
                Vssr = 0
        else:
            Vssr = 0
        return Vssr
    
    def Vbr(self, p, q):
        if self.n0row == self.n0column:
            if self.nplsrow == self.nplscolumn + 1 and self.nmnsrow == self.nmnscolumn + 1:
                Vbr = (2*np.sqrt((self.nplscolumn + 1)*(self.nmnscolumn + 1))
                       *((-1)**(self.mBcolumn - self.kcolumn))*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.symbol(p, q))
            elif  self.nplsrow == self.nplscolumn - 1 and self.nmnsrow == self.nmnscolumn - 1:
                Vbr = (2*np.sqrt((self.nplscolumn)*(self.nmnscolumn))
                       *((-1)**(self.mBcolumn - self.kcolumn))*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.symbol(p, q))
            elif  self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnscolumn:
                Vbr = (2*(self.nplscolumn + self.nmnscolumn + 1)
                       *((-1)**(self.mBcolumn - self.kcolumn))*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.symbol(p, q))
            else:
                Vbr = 0
        else:
            Vbr = 0
        return Vbr
    
    def angular_coupling(self, p, q, m_couple):
        if m_couple == + 1:
            if self.nplsrow - self.nplscolumn == + 1 and self.nmnsrow - self.nmnscolumn == 0:
                Vam =  (np.sqrt(self.nplscolumn + 1) 
                        *(-1)**(self.mBcolumn - self.kcolumn)*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.symbol_couple(p, q, - m_couple))
            elif self.nplsrow - self.nplscolumn == 0 and self.nmnsrow - self.nmnscolumn == - 1:
                Vam =  (np.sqrt(self.nmnscolumn) 
                        *(-1)**(self.mBcolumn - self.kcolumn)*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.symbol_couple(p, q, - m_couple))
            else:
                Vam = 0
        elif m_couple == - 1:
            if self.nplsrow - self.nplscolumn == - 1 and self.nmnsrow - self.nmnscolumn == 0:
                Vam = (np.sqrt(self.nplscolumn) 
                       *(-1)**(self.mBcolumn - self.kcolumn)*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.symbol_couple(p, q, - m_couple))
            elif self.nplsrow - self.nplscolumn == 0 and self.nmnsrow - self.nmnscolumn == + 1:
                Vam = (np.sqrt(self.nmnscolumn + 1) 
                       *(-1)**(self.mBcolumn - self.kcolumn)*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.symbol_couple(p, q, - m_couple))
            else:
                Vam = 0
        else:
            print("!! ERROR, m_couple is not 1 !!")
        return Vam
    
    def Vam(self, p, q, m_couple):
        if self.n0row == self.n0column:
            Vam = self.angular_coupling(p, q, m_couple)
        else:
            Vam = 0
        return Vam
    
    def VamR(self, p, q, m_couple):
        #VamR
        VVamR = 0
        for i in range(0, n0 + 1):
            VVamR += (Ri[i] + Re)**(-nn)*Ri_vec[self.n0row, i]*Ri_vec[self.n0column, i]
        VamR = VVamR*self.angular_coupling(p, q, m_couple)
        return VamR

class SymmetrizeMatrix(MatrixElement):

    def projectA1(self):
        if self.krow%2 == 0:
            if self.jrow == self.jcolumn and self.n0row == self.n0column and self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnsrow and self.mBrow == self.mBcolumn:
                if self.krow == 0 and self.kcolumn == 0:
                    projectA1 = (+ 1
                                 + (-1)**self.jrow*(-1)**(self.krow/2)
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*(self.kcolumn + self.krow)/4)
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*(self.kcolumn + self.krow)/4)*(-1)**self.jrow*(-1)**(self.kcolumn/2)
                                 )
                elif self.krow == self.kcolumn:
                    projectA1 = (+ 1
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*(self.kcolumn + self.krow)/4)
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*(- self.kcolumn + self.krow)/4)*(-1)**self.jrow*(-1)**(- self.kcolumn/2)
                                 )
                elif self.krow == - self.kcolumn:
                    projectA1 = (+ (-1)**self.jrow*(-1)**(self.krow/2)
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*(self.kcolumn + self.krow)/4)
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*(- self.kcolumn + self.krow)/4)*(-1)**self.jrow*(-1)**(- self.kcolumn/2)
                                 )
                #+k'と-k'のダブルカウントする必要がある。
                elif self.kcolumn%2 == 0:
                    projectA1 =(+ 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*(self.kcolumn + self.krow)/4)
                                + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*(- self.kcolumn + self.krow)/4)*(-1)**self.jrow*(-1)**(- self.kcolumn/2)
                                )
                else:
                    projectA1 = 0
            else:
                projectA1 = 0
        else:
            projectA1 = 0
        return round(projectA1, 10)
    
    def projectE1(self):
        if self.krow%2 == 0:
            if self.jrow == self.jcolumn and self.n0row == self.n0column and self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnsrow and self.mBrow == self.mBcolumn:
                if self.krow == 0 and self.kcolumn == 0:
                    projectE1 = (+ 1 
                                 + (-1)**self.jrow*(-1)**(self.krow/2) 
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((self.kcolumn + self.krow)/4 - 2/3))
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((self.kcolumn + self.krow)/4 - 2/3))*(-1)**self.jrow*(-1)**(self.kcolumn/2)
                                 )
                elif self.krow == self.kcolumn:
                    projectE1 = (+ 1
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((self.kcolumn + self.krow)/4 - 2/3))
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((- self.kcolumn + self.krow)/4 - 2/3))*(-1)**self.jrow*(-1)**(- self.kcolumn/2)
                                 )
                elif self.krow == - self.kcolumn:
                    projectE1 = (+ (-1)**self.jrow*(-1)**(self.krow/2)
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((self.kcolumn + self.krow)/4 - 2/3))
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((- self.kcolumn + self.krow)/4 - 2/3))*(-1)**self.jrow*(-1)**(- self.kcolumn/2)
                                 )
                #+k'と-k'のダブルカウントする必要がある。
                elif self.kcolumn%2 == 0:
                    projectE1 = (+ 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((self.kcolumn + self.krow)/4 - 2/3))
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((- self.kcolumn + self.krow)/4 - 2/3))*(-1)**self.jrow*(-1)**(- self.kcolumn/2)
                                 )
                else:
                    projectE1 = 0
            else:
                projectE1 = 0
        else:
            projectE1 = 0
        return round(projectE1, 10)
    
    def projectE2(self):
        if self.krow%2 == 0:
            if self.jrow == self.jcolumn and self.n0row == self.n0column and self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnsrow and self.mBrow == self.mBcolumn:
                if self.krow == 0 and self.kcolumn == 0:
                    projectE2 = (+ 1
                                 + (-1)**self.jrow*(-1)**(self.krow/2)
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((self.kcolumn + self.krow)/4 + 2/3))
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((self.kcolumn + self.krow)/4 + 2/3))*(-1)**self.jrow*(-1)**(self.kcolumn/2)
                                 )
                elif self.krow == self.kcolumn:
                    projectE2 = (+ 1
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((self.kcolumn + self.krow)/4 + 2/3))
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((- self.kcolumn + self.krow)/4 + 2/3))*(-1)**self.jrow*(-1)**(- self.kcolumn/2)
                                 )
                elif self.krow == - self.kcolumn:
                    projectE2 = (+ (-1)**self.jrow*(-1)**(self.krow/2)
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((self.kcolumn + self.krow)/4 + 2/3))
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((- self.kcolumn + self.krow)/4 + 2/3))*(-1)**self.jrow*(-1)**(- self.kcolumn/2)
                                 )
                #+k'と-k'のダブルカウントする必要がある。
                elif self.kcolumn%2 == 0:
                    projectE2 = (+ 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((self.kcolumn + self.krow)/4 + 2/3))
                                 + 2*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.cos(np.pi*((- self.kcolumn + self.krow)/4 + 2/3))*(-1)**self.jrow*(-1)**(- self.kcolumn/2)
                                 )
                else:
                    projectE2 = 0
            else:
                projectE2 = 0
        else:
            projectE2 = 0
        return round(projectE2, 10)
    
    def projectF1(self):
        if self.krow%2 == 0:
            if self.jrow == self.jcolumn and self.n0row == self.n0column and self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnsrow and self.mBrow == self.mBcolumn:
                if self.krow == 0 and self.kcolumn == 0:
                    projectF1 = 1 - (-1)**self.jrow*(-1)**(self.krow/2)
                elif self.krow == self.kcolumn:
                    projectF1 = 1
                elif self.krow == - self.kcolumn:
                    projectF1 = - (-1)**self.jrow*(-1)**(self.krow/2)
                else:
                    projectF1 = 0
            else:
                projectF1 = 0
        else:
            projectF1 = 0
        return round(projectF1, 10)
    
    def projectF2(self):
        if self.krow%2 == 1:
            if self.jrow == self.jcolumn and self.n0row == self.n0column and self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnsrow and self.mBrow == self.mBcolumn:
                #+k'と-k'のダブルカウントする必要がある。
                if self.kcolumn%2 == 1:
                    projectF2 = (+ (- 1)**(self.kcolumn + self.krow)*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.exp(np.pi*1j*(self.kcolumn + self.krow)/4)
                                 + (- 1)**(- self.kcolumn + self.krow)*wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.exp(np.pi*1j*(- self.kcolumn + self.krow)/4)*1j*(-1)**self.jrow*(-1)**((- self.kcolumn - 1)/2)
                                 )
                    #projectF2 = round(projectF2, 10)
                    
                else:
                    projectF2 = 0
            else:
                projectF2 = 0
        else:
            projectF2 = 0
        return projectF2
    
    def projectF3(self):
        if self.krow%2 == 1:
            if self.jrow == self.jcolumn and self.n0row == self.n0column and self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnsrow and self.mBrow == self.mBcolumn:
                #+k'と-k'のダブルカウントする必要がある。
                if self.kcolumn%2 == 1:
                    projectF3 = (+ wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow + self.kcolumn][self.jrow + self.krow]*np.exp(- np.pi*1j*(self.kcolumn + self.krow)/4)
                                 + wigner_d_small_pihalf.wigner_d_small_pihalf(self.jrow)[self.jrow - self.kcolumn][self.jrow + self.krow]*np.exp(- np.pi*1j*(- self.kcolumn + self.krow)/4)*(- 1j)*(-1)**self.jrow*(-1)**((- self.kcolumn - 1)/2)
                                 )
                    #projectF3 = round(projectF3, 10)
                else:
                    projectF3 = 0
            else:
                projectF3 = 0
        else:
            projectF3 = 0
        return projectF3

def H_H(m):
    dim = MatrixParameter(m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).size()[0]
    print(dim)
    # 時間計測開始
    time_sta = time.time()
    
    result = np.array([[+ MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Ts() *shbaromega/4
                        + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vs() 
                        
                        + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Tb() *bhbaromega/4
                        + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vb() *bhbaromega/4
                        
                        + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Tr() *B
                        
                        + V3*(
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vr(3, 2)
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vr(3, -2) )
                        
                        + V4*(
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vr(4, 0) *np.sqrt(14)
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vr(4, 4) *(- np.sqrt(5))
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vr(4, -4) *(- np.sqrt(5)) )
                        
                        + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vsb() *bhbaromega/(4*kxx)
                        
                        + VR3*np.sqrt(shbaromega/(4*kzz*az**2))*(
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vsr(3, 2)
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vsr(3, -2) ) 
                        
                        + VR4*np.sqrt(shbaromega/(4*kzz*az**2))*(
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vsr(4, 0) *np.sqrt(14)
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vsr(4, 4) *(- np.sqrt(5))
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vsr(4, -4) *(- np.sqrt(5)) )
                        
                        + VRR3*(shbaromega/(4*kzz*az**2))*(
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vssr(3, 2)
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vssr(3, -2) )
                        
                        + VRR4*(shbaromega/(4*kzz*az**2))*(
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vssr(4, 0) *np.sqrt(14)
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vssr(4, 4) *(- np.sqrt(5))
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vssr(4, -4) *(- np.sqrt(5)) )
                        
                        + Vrho3*bhbaromega/(4*kxx)*(
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vbr(3, 2)
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vbr(3, -2) )
                        
                        + Vrho4*bhbaromega/(4*kxx)*(
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vbr(4, 0) *np.sqrt(14)
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vbr(4, 4) *(- np.sqrt(5))
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vbr(4, -4) *(- np.sqrt(5)) )
                        
                        + Vam31*bhbaromega/(4*kxx)*(
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vam(3, + 2, + 1)
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vam(3, + 2, - 1) 
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vam(3, - 2, + 1)
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).Vam(3, - 2, - 1) )
                        
                        + VamR31*bhbaromega/(4*kxx)*(
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).VamR(3, + 2, + 1)
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).VamR(3, + 2, - 1) 
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).VamR(3, - 2, + 1)
                            + MatrixElement(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).VamR(3, - 2, - 1) ) 

               for n0column in range(0, n0 + 1)
               for mBcolumn in range(- j, j + 1)
               #for lcolumn in range(- nmns, npls + 1 , +1)  #for lcolumn, if mBcolumn + lcolumn == m ->  lcolumn = m - mBcolumn 
               for vcolumn in range(abs(m - mBcolumn), npls + nmns - abs(m - mBcolumn) + 2, 2)   #lcolumn = m - mBcolumn 
               for jcolumn in range(abs(mBcolumn), j + 1)
               for kcolumn in range(- jcolumn, jcolumn + 1)
               #if mBcolumn + lcolumn == m
                  ]
                  for n0row in range(0, n0 + 1)
                  for mBrow in range(- j, j + 1) 
                  #for lrow in range(- nmns, npls + 1 , +1)  #for lrow, if mBrow + lrow == m ->  lrow = m - mBrow 
                  for vrow in range(abs(m - mBrow), npls + nmns - abs(m - mBrow) + 2, 2) #lrow = m - mBrow 
                  for jrow in range(abs(mBrow), j + 1)
                  for krow in range(-jrow, jrow + 1)
                  #if mBrow + lrow == m
                  ])
    result = result.astype(np.float64)
    
    # 時間計測終了
    time_end = time.time()
    # 経過時間（秒）
    global tim
    tim = time_end- time_sta
    print('matrix time: ',tim)

    #print("new, \n", np.array(result))
    
    if len(result) != dim:
        print("!! dimension error !!")
    return result

def projection(m, symmetry):
    # 時間計測開始
    time_sta_sym = time.time()
    result = np.array([[+ SymmetrizeMatrix(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).projectA1() if symmetry == 'A1' else
                        + SymmetrizeMatrix(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).projectE1() if symmetry == 'E1' else
                        + SymmetrizeMatrix(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).projectE2() if symmetry == 'E2' else
                        + SymmetrizeMatrix(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).projectF1() if symmetry == 'F1' else
                        + SymmetrizeMatrix(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).projectF2() if symmetry == 'F2' else
                        + SymmetrizeMatrix(m, jrow, mBrow, krow, n0row, vrow, jcolumn, mBcolumn, kcolumn, n0column, vcolumn).projectF3() if symmetry == 'F3' else
                        print("!!symmetry input error !!")
                            for n0column in range(0, n0 + 1)
                            for mBcolumn in range(- j, j + 1)
                            for vcolumn in range(abs(m - mBcolumn), npls + nmns - abs(m - mBcolumn) + 2, 2)   #lcolumn = m - mBcolumn 
                            for jcolumn in range(abs(mBcolumn), j + 1)
                            for kcolumn in range(- jcolumn, jcolumn + 1)
                            ]
                                for n0row in range(0, n0 + 1)
                                for mBrow in range(- j, j + 1) 
                                for vrow in range(abs(m - mBrow), npls + nmns - abs(m - mBrow) + 2, 2) #lrow = m - mBrow 
                                for jrow in range(abs(mBrow), j + 1)
                                for krow in range(-jrow, jrow + 1)
                                ])
    if symmetry == 'A1' or symmetry == 'E1' or symmetry == 'E2' or symmetry == 'F1':
        result = result.astype(np.float64)
    if symmetry == 'F2' or symmetry == 'F3':
        result = result.astype(np.complex128)
    
    # 時間計測終了
    time_end_sym = time.time()
    # 経過時間（秒）
    global tim_sym
    tim_sym = time_end_sym - time_sta_sym
    print(symmetry, 'matrix time: ',tim_sym)
    dim = MatrixParameter(m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).size()[0]
    if len(result) != dim:
        print("!! dimension error !!")
    return result

def diagonalize(H, U, symmetry):
    # 時間計測開始
    time_sta_diag = time.time()
    
    m = calcm
    f = open(symmetry + '_titleMorse.txt', 'w')
    f.write('j =' + str(j) + '\n')
    f.write('n0 =' + str(n0) + '\n')
    f.write('npls, nmns =' + str(npls) + ',' + str(nmns) + '\n')
    f.write('parameter' + '(kzz, kxx, kxxz) =' + '(' + str(kzz) + ',' + str(kxx) +','+ str(kxxz)  +')' + '\n')
    f.write('parameter' + '(V3, V4, VR3, VR4, Vrho3, Vrho4, Vam31) =' + '(' + ',' + str(V3) +','+ str(V4) + ',' + str(VR3) +','+ str(VR4) + str(Vrho3) +','+ str(Vrho4) + str(Vam31) +')' + '\n')
    g = open(symmetry + '_titleMorse_vector.txt', 'w')
    g.write('j =' + str(j) + '\n')
    g.write('n0 =' + str(n0) + '\n')
    g.write('npls, nmns =' + str(npls) + ',' + str(nmns) + '\n')
    g.write('parameter' + '(kzz, kxx, kxxz) =' + '(' + str(kzz) + ',' + str(kxx) +','+ str(kxxz)  +')' + '\n')
    g.write('parameter' + '(V3, V4, VR3, VR4, Vrho3, Vrho4, Vam31) =' + '(' + ',' + str(V3) +','+ str(V4) + ',' + str(VR3) +','+ str(VR4) + str(Vrho3) +','+ str(Vrho4) + str(Vam31) +')' + '\n')
    
    #diagonalize
    eig_val,eigen_vector = np.linalg.eigh(H)
    print('m =', m)       
    np.set_printoptions(threshold=np.inf)
    f.write( 'm =' + str(m) +  '\n')
    np.set_printoptions(threshold=np.inf)
    f.write( 'eigen value\n{}\n'.format(eig_val))
    eigen_vector = np.dot(np.conjugate(U.T), eigen_vector)
    eigen_vector = np.transpose(eigen_vector)
    g.write('\n' + 'm =' + str(m) + '\n')
    g.write('eigen value' + ',' + 'eigen vector' + '(j, k, m, n0)' + '\n')
    
    dim = []
    jmdim = []
    allowl = []
    allowmB = []
        
    for mB in range(- j - 1, j):
        mB += 1
        for ll in range(- nmns - 1, npls):
            ll += 1
            jdim = []
            for num in range(- 1,j):
                num += 1
                jd = (2 * abs(num) + 1)
                jdim.append(jd)
            
            mminus = []
            for mm in range( - 1, abs(mB)):
                mm += 1
                if mm == 0:
                    a = 0
                else:
                    a = (2 * (abs(mm) - 1) + 1)
                mminus.append(a)
            
            if ll + mB == m:
                d = (npls + 1 - abs(ll))*(sum(jdim)-sum(mminus))
                dim.append(d)
                jmdim.append(sum(jdim)-sum(mminus))
                allowl.append(ll)
                allowmB.append(mB)
    jcount = []
    jjcount = []
    j2count = []
    for ja in range(-1, j):
        ja += 1
        jcount.append(ja)
        jb = (2*ja + 1)
        jjcount.append(jb)
        jc = sum(jjcount)
        j2count.append(jc)
    #eigen vector is columnn one.
    for r in range(0, len(eigen_vector)):
        for c in range(0, len(eigen_vector[r])):
            #select eigen vectors larger contribution than 0.1 
            if abs(eigen_vector[r, c]) > diag_coeff:
                
                #quantum number n0
                quantn0 = c // sum(dim)
                #quantum number m
                quantm = m
                
                #csurp:n0で割ったやつ、すべての(mB, l)行列通しての番号
                csurp = c % sum(dim)
                
                sumd = [0]
                #quantum number npls, nmns, mB
                for d in range(- 1, len(dim) - 1):
                    d += 1
                    sumd.append(sumd[d] + dim[d])
                    #print(dim)
                    #if d == 0 and csurp < dim[d]:
                    if sumd[d] <= csurp and csurp < sumd[d + 1]:
                        quantl = allowl[d]
                        quantmB = allowmB[d]
                        
                        #jsurp:各(mB, l)行列での番号
                        jsurp = csurp - sumd[d]

                        if quantmB == 0:
                            quantv = 2*(jsurp // sum(jjcount)) + abs(quantl)
                            ksurp = jsurp % j2count[j]
                        elif j == 0:
                            quantv = 2*(jsurp // sum(jjcount)) + abs(quantl)
                            ksurp = jsurp % j2count[j]
                        else:
                            quantv = 2*(jsurp // (sum(jjcount) - j2count[abs(quantmB) - 1])) + abs(quantl)
                            ksurp = jsurp % (j2count[j] - j2count[abs(quantmB) - 1]) + j2count[abs(quantmB) - 1]

                        if ksurp == 0:
                            quantj = abs(quantmB) 
                            quantk = quantmB

                            g.write(str(round(eig_val[r], 10)) + ',' + str(np.round(eigen_vector[r, c], 5))  + ',' + str(quantn0) + ','  + str(quantv) + ',' + str(quantl)+ ',' + str(quantj) + ',' + str(quantk) + ',' + str(quantmB) + ',' + str(quantm) + ',' + '\n')
                        for jc in range(-1, j):
                            jc += 1                            
                            if ksurp >= j2count[jc] and ksurp < j2count[jc + 1]:
                                #このとき、jcがjに対応、j2count(jc)がkの情報をもつ
                                quantj = jc + 1
                                quantk = ksurp - j2count[jc] - quantj
                                g.write(str(round(eig_val[r], 10)) + ',' + str(np.round(eigen_vector[r, c], 5))  + ',' + str(quantn0) + ','  + str(quantv) + ',' + str(quantl)+ ',' + str(quantj) + ',' + str(quantk) + ',' + str(quantmB) + ',' + str(quantm) + ',' + '\n')

    # 時間計測終了
    time_end_diag = time.time()
    # 経過時間（秒）
    global tim_diag    
    tim_diag = time_end_diag - time_sta_diag
    print('diagonalizing time: ',tim_diag)

def pre_symmetrized_diagonalize(m, H):
    
    A1 = projection(m, 'A1')
    E1 = projection(m, 'E1')
    E2 = projection(m, 'E2')
    F1 = projection(m, 'F1')
    F2 = projection(m, 'F2')
    F3 = projection(m, 'F3')
    projects = (A1, E1, E2, F1, F2, F3)
    
    p = 0
    dimension = MatrixParameter(m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).size()[0]
    
    projectA1 = np.empty((0, dimension), int)
    projectE1 = np.empty((0, dimension), int)
    projectE2 = np.empty((0, dimension), int)
    projectF1 = np.empty((0, dimension), int)
    projectF2 = np.empty((0, dimension), int)
    projectF3 = np.empty((0, dimension), int)
    
    projectA = np.empty((0, dimension), int)
    projectE = np.empty((0, dimension), int)
    projectF = np.empty((0, dimension), int)
    sym = np.empty((0, dimension), int)
    
    for project in projects:
        p += 1
        project = project[np.any(project != 0, axis = 1)]    #ゼロベクトルの削除
        for rr in range (0, project.shape[0]):
            num = 0
            maxindex = 0
            #固有ベクトル(各行(row))の規格化
            for cc in range (0, project.shape[1]):
                num += abs(project[rr, cc])**2
            maxindex = np.argmax(abs(project[rr, :]))
            if p == 2:
                for cc in range (0, project.shape[1]):
                    print(p, rr, maxindex, project[rr, cc]/np.sqrt(num)/(project[rr, maxindex]/abs(project[rr, maxindex])), project[rr, cc], np.sqrt(num)/(project[rr, maxindex]/abs(project[rr, maxindex])), project[rr, maxindex], abs(project[rr, maxindex]), (project[rr, maxindex]/abs(project[rr, maxindex])))
            project[rr, :] = project[rr, :]/np.sqrt(num)/(project[rr, maxindex]/abs(project[rr, maxindex]))    #maxelementで各ベクトルの位相合わせ(最大の係数が正の実数となるよ
            #print(project[rr, :])
        #print(np.round(project, 3))
        project = np.round(list(project), 10)

        #重複する固有ベクトルを削除
        #Eはまぜかnp.uniqueだとうまく重複するベクトルが削除できなかったので自作
        #重複するベクトルが連続している場合のみ使える仕様にしている。A,Fはそうではないので使えない
        if p == 1 or p == 2 or p == 3: #float用の評価式　#j = 7まで
            samerow = []
            for rrr in range(0, len(project)):
                for rrrr in range(0, len(project)):
                    #if rrr == 25 and rrrr == 28:
                    #    print(np.round(project[rrr], 3))
                    #    print(np.round(project[rrrr], 3))
                    if all([int(project[rrrr][cc] - project[rrr][cc] < 10**(-6)) == 1  for cc in range(0, len(project[rrrr]))]):
                        if rrr < rrrr:  #rrr, rrrr入れ替わりによるカウントの重複阻止
                            samerow.append(rrrr)

            samerow = np.unique(samerow)    #重複する行番号を削除 complex, float配列はうまくできなくてもintならうまくいきそう
            print("samerow, ", samerow)


            #samerowの行を消す
            samerow_length = len(samerow)
            for compnent in range(0, samerow_length):
                if len(samerow) != 0:
                    project = np.delete(project, samerow[len(samerow) - 1], 0)
                    samerow = np.delete(samerow, len(samerow) - 1)
        
        project_array, index = np.unique(project, return_index = True, axis = 0)    #重複する固有ベクトルを削除
        index = sorted(index)
        project_list = []
        for idx in index:
            project_list.append(project[idx])
        project_nonzero = np.array(project_list)    #固有ベクトルをもとの順番に並べ変え    #project_nonzeroは非ゼロ成分のみを持った横長の行列
        
        
        if len(project_nonzero) != 0:
            sym = np.append(sym, project_nonzero, axis = 0)    #symはA, E, F全ての対称性の非ゼロベクトルを並べたユニタリー行列
            if p == 1:
                projectA1 = np.append(projectA1, project_nonzero, axis = 0)
                projectA = np.append(projectA, project_nonzero, axis = 0)
            elif p == 2:
                projectE1 = np.append(projectE1, project_nonzero, axis = 0)
                projectE = np.append(projectE, project_nonzero, axis = 0)
            elif p == 3:
                projectE2 = np.append(projectE2, project_nonzero, axis = 0)
                projectE = np.append(projectE, project_nonzero, axis = 0)
            elif p == 4:
                projectF1 = np.append(projectF1, project_nonzero, axis = 0)
                projectF = np.append(projectF, project_nonzero, axis = 0)
            elif p == 5:
                projectF2 = np.append(projectF2, project_nonzero, axis = 0)
                projectF = np.append(projectF, project_nonzero, axis = 0)
            elif p == 6:
                projectF3 = np.append(projectF3, project_nonzero, axis = 0)
                projectF = np.append(projectF, project_nonzero, axis = 0)
        else:
            pass

    projects6 = [projectA1, projectE1, projectE2, projectF1, projectF2, projectF3]
    projects3 = [projectA, projectE, projectF]
    return projects6, projects3, sym

def symmetrized_diagonalize6(m, H):
    q = 0
    s = 0
    projects = pre_symmetrized_diagonalize(m, H)[0]
    dimension = MatrixParameter(m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).size()[0]
    zero = np.zeros((1, dimension))
    for project in projects:
        q += 1
        project_nonzero = project
        dim_nonzero = len(project)
        print(dim_nonzero)
        if dim_nonzero != 0:
            for s in range(dimension - dim_nonzero):
                project = np.append(project, zero, axis = 0)    #正方行列になるようにゼロフィル　　　　＃project_squareは下の成分をゼロフィルした正方行列
                project_square = project
        else:
            project_square = np.zeros((dimension, dimension))
        #np.dotは順序が逆
        H_Gamma_m = np.dot(project_square, np.dot(H, np.conjugate(project_square.T)))
        H_Gamma_m = np.delete(H_Gamma_m, slice(dim_nonzero, dimension), 0)
        H_Gamma_m = np.delete(H_Gamma_m, slice(dim_nonzero, dimension), 1)
        #print("Hamiltonian", "\n", H_Gamma_m)

        #print([len(v) for v in project_nonzero])
        if len(project_nonzero)!= 0 and (q == 1 or q == 2 or q == 4):   #A1, E1,F1のみ対角化。他は同じ固有値を与える。

            #diagonalize
            diagonalize(H_Gamma_m, project_nonzero)  

            
def symmetrized_diagonalize3(m, H):
    name = 0
    project_name = ['A', 'E', 'F']
    projects = pre_symmetrized_diagonalize(m, H)[1]
    dimension = MatrixParameter(m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).size()[0]
    zero = np.zeros((1, dimension))
    for project in projects:
        project_nonzero = project
        dim_nonzero = len(project)
        print(dim_nonzero)
        project_square = np.identity(dimension)
        if dim_nonzero != 0:
            for s in range(dimension - dim_nonzero):
                project = np.append(project, zero, axis = 0)    #正方行列になるようにゼロフィル　　　　＃project_squareは下の成分をゼロフィルした正方行列
                project_square = project
        else:
            project_square = np.identity(dimension)
        #np.dotは順序が逆
        #print(np.round(project_square, 1))
        H_Gamma_m = np.dot(project_square, np.dot(H, np.conjugate(project_square.T)))
        H_Gamma_m = np.delete(H_Gamma_m, slice(dim_nonzero, dimension), 0)
        H_Gamma_m = np.delete(H_Gamma_m, slice(dim_nonzero, dimension), 1)
        #print("Hamiltonian", "\n", H_Gamma_m)
        
        #print([len(v) for v in project_nonzero])
        if len(project_nonzero)!= 0:
            #diagonalize
            diagonalize(H_Gamma_m, project_nonzero, project_name[name])
            name += 1

def symmetrized_diagonalize1(m, H):
    project_square = pre_symmetrized_diagonalize(m, H)[2]
    #print(np.round(np.dot(np.conjugate(project_square.T), project_square).real, 3))
    H_Gamma_m = np.dot(project_square, np.dot(H, np.conjugate(project_square.T)))
    
    #diagonalize
    diagonalize(H_Gamma_m, project_square)

def main ():
    # 時間計測開始
    time_sta_whole = time.time()
    
    Hamiltonian = H_H(calcm)    #Hamiltonian matrix

    diagonalize(Hamiltonian, np.identity(len(Hamiltonian)), '') #diagonalization, 基底対称化を行わない場合
    #symmetrized_diagonalize6(calcm, Hamiltonian) #diagonalization, 基底対称化を行ってA1, E1, F1を対角化　固有ベクトルを必要としない場合
    #symmetrized_diagonalize3(calcm, Hamiltonian) #diagonalization, 基底対称化を行ってA, E, Fを対角化　固有ベクトルが必要な場合
    #symmetrized_diagonalize1(calcm, Hamiltonian) #diagonalization, 基底対称化を行ったユニタリーなブロック対角行列を対角化, テスト用
    
    # 時間計測終了
    time_end_whole = time.time()
    # 経過時間（秒）   
    tim_whole = time_end_whole - time_sta_whole
    print('whole time: ',tim_whole)
    
    f = open('run_time.txt', 'w')
    f.write('j = ' + str(j) + '\n')
    f.write('n0 = ' + str(n0) + '\n')
    f.write('npls, nmns = ' + str(npls) + ',' + str(nmns) + '\n')
    f.write('matrix time = ' + str(tim) + '\n')
    #f.write('F3 matrix time = ' + str(tim_sym) + '\n')
    f.write('diagonalizing time = ' + str(tim_diag) + '\n')
    f.write('whole time = ' + str(tim_whole) + '\n')
    return 0
    
print(main())
