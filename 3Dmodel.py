# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:12:08 2021

@author: sasak
"""

import scipy.linalg
import numpy as np
from sympy.physics.wigner import wigner_3j

#考慮する角運動量の上限 
j = 10
B = 5.241
#ポテンシャルパラメータ
(V3, V4, V6, V7) = (60, 0, 0, 0)
        
def C_m(m, p, q):
    mcount = m        
    #適切な次元の行列の作成
    dim = []
    for num in range(-1,j):
        num += 1
        jdim = (2 * num + 1)
        dim.append(jdim)
    mminus = []
    #only positive mcount
    for mm in range(-1, mcount):
        mm += 1
        if mm == 0:
            a = 0
        else:
            a = (2 * (mm - 1) + 1)
        mminus.append(a)
    result = np.ones((sum(dim)-sum(mminus), sum(dim)-sum(mminus)))

    #行の指定(j)
    jrsum = []
    for jrow in range(mcount - 1,j):
        jrow += 1
        if jrow == mcount:
            jr = 0
        else:
            jr = (2 * (jrow - 1) + 1)
        jrsum.append(jr)
            
        #行の指定(k)
        krsum = []
        for krow in range(-jrow - 1,jrow):
            krow += 1
            kr = 1
            krsum.append(kr)
            
            #列の指定(j)
            jcsum = []
            for jcolumn in range(mcount - 1,j):
                jcolumn += 1
                if jcolumn == mcount:
                    jc = 0
                else:
                    jc = (2 * (jcolumn - 1) + 1)
                jcsum.append(jc) 
                
                #列の指定(k)
                kcsum = []
                for kcolumn in range(-jcolumn-1,jcolumn):
                    kcolumn += 1
                    kc = 1
                    kcsum.append(kc)
                    
                    #2つの3-j symbolの定義
                    symbol1 = wigner_3j(jrow, p, jcolumn, -krow, q, kcolumn)
                    symbol2 = wigner_3j(jrow, p, jcolumn, -mcount, 0, mcount)
                    symbol = symbol1*symbol2
                    mat_el = ((-1)**(mcount - kcolumn))*np.sqrt((2*jcolumn + 1)*(2*jrow + 1))*symbol
                    row = sum(jrsum) + (len(krsum) - 1)
                    column = sum(jcsum) + (len(kcsum) - 1)
                    result[row,column] = mat_el
                    
    return result

def Hrot_m(m):        
    #適切な次元の行列の作成
    dim = []
    for num in range(-1,j):
        num += 1
        jdim = (2 * num + 1)
        dim.append(jdim)
    mminus = []
    #only positive m
    for mm in range(-1, m):
        mm += 1
        if mm == 0:
            a = 0
        else:
            a = (2 * (mm - 1) + 1)
        mminus.append(a)
    result = np.ones((sum(dim)-sum(mminus), sum(dim)-sum(mminus)))

    #行の指定(j)
    jrsum = []
    for jrow in range(m - 1,j):
        jrow += 1
        if jrow == m:
            jr = 0
        else:
            jr = (2 * (jrow - 1) + 1)
        jrsum.append(jr)
            
        #行の指定(k)
        krsum = []
        for krow in range(-jrow - 1,jrow):
            krow += 1
            kr = 1
            krsum.append(kr)
            
            #列の指定(j)                
            jcsum = []
            for jcolumn in range(m - 1,j):
                jcolumn += 1
                if jcolumn == m:
                    jc = 0
                else:
                    jc = (2 * (jcolumn - 1) + 1)
                jcsum.append(jc) 
                
                #列の指定(k)
                kcsum = []
                for kcolumn in range(-jcolumn-1,jcolumn):
                    kcolumn += 1
                    kc = 1
                    kcsum.append(kc)
                    
                    #Hamiltonianは既に対角化されている
                    if jrow == jcolumn and krow == kcolumn: 
                        mat_el = B*jrow*(jrow + 1)
                    else: 
                        mat_el = 0
                    #座標の設定
                    row = sum(jrsum) + (len(krsum) - 1)
                    column = sum(jcsum) + (len(kcsum) - 1)
                    result[row, column] = mat_el

    return result


#mについて総和を取る用           
def diagonalize():
    print('j =', j)
    print('parameter','(V3, V4, V6, V7) =', (V3, V4, V6, V7))
    f = open('title.txt', 'w')
    f.write('j =' + str(j) + '\n')
    f.write('parameter' + '(V3, V4) =' + '(' + str(V3) +','+ str(V4) + str(V6) +','+ str(V7) +')' + '\n')
    #only positive m    
    for m in range(-1, j):
        m += 1
        #ポテンシャルVの定義(エネルギーはB^(CH_4)で規格化)           
        T3_m = C_m(m, 3, 2) + C_m(m, 3, -2)
        T4_m = np.sqrt(14)*C_m(m, 4, 0) - np.sqrt(5)*C_m(m, 4, 4) - np.sqrt(5)*C_m(m, 4, -4)
        T6 = 2*C_m(m, 6, 0) + np.sqrt(14)*(C_m(m, 6, 4) + C_m(m, 6, -4))
        T7 = -np.sqrt(11)*(C_m(m, 7, 6) + C_m(m, 7, -6)) + np.sqrt(13)*(C_m(m, 7, 2) + C_m(m, 7, -2))
        V_m = V3*T3_m + V4*T4_m + V6*T6 + V7*T7
        H_m = Hrot_m(m) + V_m
        
        #対角化
        lo = 0
        hi = 1
        #eig_val,eigen_vector = scipy.linalg.eigh(H_m,eigvals=(lo,hi))
        eig_val,eigen_vector = scipy.linalg.eigh(H_m)
        
        #固有値を表示 
        np.set_printoptions(threshold=np.inf)
        print('m =', m)       
        #print(H_m)
        print('eigen value\n{}\n'.format(eig_val))
        f.write( 'm =' + str(m) +  '\n')
        f.write( 'eigen value\n{}\n'.format(eig_val))
        

    return


print(diagonalize())                                 