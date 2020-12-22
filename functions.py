from math import sqrt
import numpy as np

def decimalator(hex_number):
    return(int(hex_number,16))

def binalinator(integer):
    if integer < 0:
        return '-' + bin(integer)[3:].zfill(32)
    else:
        return bin(integer)[2:].zfill(32)
    
def Hash(hex_string):
    integer = decimalator(hex_string)
    function = integer % 2**32
    return function

def position_leftmost_1(bin_number):
    count=0
    for i in bin_number:
        count += 1
        if i=="1":
            break
    return count   

def address(string,b):
    return string[:b]          

def remaining(string,b):
    return string[b:]

def Hyperloglog(m,b,binlist):
    HLL=np.zeros(2**b)
    for i in range(len(binlist)):
        string=binlist[i]
        j= int(address(string,b),2)
        w=position_leftmost_1(remaining(string,b))
        HLL[j]=max(HLL[j],w)
    
    return  HLL

 
def cardinality_error(HLL,m):
    a_m=0.7213/(1+1.079/(m))
    Z=1/sum([2**(-bucket) for bucket in HLL])
    cardinality=a_m*(m**2)*Z
    error=(1.04/sqrt(m))
    
    return cardinality,error



    