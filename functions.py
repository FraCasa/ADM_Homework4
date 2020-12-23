from math import sqrt
import numpy as np
import pandas as pd 
import matplotlib.pyplot  as plt
import statistics
from tqdm import tqdm
import seaborn as sns
import random

from sklearn.cluster import KMeans
import spacy
import time
from langdetect import detect
import nltk
import math
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from math import *
import random

import numpy.matlib
from sklearn.metrics.pairwise import euclidean_distances


#data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import wordcloud
from mpl_toolkits import mplot3d
import pickle

########################################QUESTION ONE ####################################################################################


def decimalator(hex_number):
    '''
    function to trasform a number from hex to dec
    input hex
    output dec
    '''
    dec_nuumber = int(hex_number,16)
    
    return dec_nuumber

def binalinator(integer):
    '''
    function to convert an int hash value to its bin value
    input: int value
    output: bin value
    '''
    if integer < 0:
        neg_bin = '-' + bin(integer)[3:].zfill(32)
        return neg_bin
    
    else:
        pos_bin = bin(integer)[2:].zfill(32)
        return pos_bin 
    
def Hash(hex_string):
    '''
    function to compute hash from a hex string 
    input: hex string
    output: hash value
    '''
    
    integer = decimalator(hex_string)
    function = integer % 2**32
    return function

def LeadingZeros(bin_list):
    '''
    function to count leading zeros
    input slice of binary number
    output counter of zeros
    '''
    count0 = 0
    #leading zero count
    for b in bin_list:

        if int(b) == 0:
            count0 +=1
            
        else:
            break            
            
    return count0

def position_leftmost_1(bin_number):
    '''
    function to count position_leftmost_1
    input slice of binary number
    output counter of zeros
    '''
        
    count=0
    for i in bin_number:
        count += 1
        if i=="1":
            break
            
    return count   



def address(string,b):
    '''
    function to extract the address part of the 32-bit binary string
    input: string, b
    output: index of the corrisponding bucket
    '''
    addr = string[:b]
    
    return addr          

def remaining(string,b):
    '''
    function to extract the part of the 32-bit binary string in which we will count the zeros
    input: string, b
    output: slice of string
    '''
    rem_part = string[b:]
    
    return rem_part

def Hyperloglog(m,b,binlist):
    '''
    HyperLogLog function
    input: m, b and a list of binary numbers
    output: hll data structure 
    '''
    
    HLL = np.zeros(2**b)
    
    for i in range(len(binlist)):
        string=binlist[i]
        j = int(address(string,b),2)
        w = position_leftmost_1(remaining(string,b)) 
        HLL[j]=max(HLL[j],w)
    
    return  HLL

def cardinality_error(HLL,m):
    '''
    Function to compute the cardinality of the dataset and error of the filter
    input: hll, m
    output: cardinality and error
    '''  
    a_m = 0.7213/(1+1.079/(m))
    
    Z = 1/sum([2**(-bucket) for bucket in HLL])
    
    cardinality = a_m*(m**2)*Z
    
    error = (1.04/sqrt(m))
    
    return cardinality,error


########################################QUESTION TWO ####################################################################################
    '''
    Takes a string in input and return a new stemmed string with no punctuation and no stopwords. 
    '''
def clean_text(plot):
    
    # Removing punctuation
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(plot.lower())
    
    # Removing stopwords
    new_plot = [word for word in tokens if not word in nltk.corpus.stopwords.words("english")]
    
    # Stemming
    stemmer = nltk.stem.porter.PorterStemmer()
    singles = [stemmer.stem(word) for word in new_plot]
    
    new_plot = ' '.join(singles)
    
    return new_plot



    '''
    this function goes through text columns and after steeming them the result will save in new columns named clean_text
    it gets data set as an inout and return data set as an output with one extra columns
    '''
def process_data(data):
    
    data['clean_text'] = data['Text'].apply(lambda x : clean_text(x))
        
    #print("Done.")

    '''
    this function will get data as an input and retun a list of all steemed words available in whole reviews as an output 

    '''
def global_lis(data):
    
    global_list = []
    process_data(data)
    
    for i in range(1,len(data)):
        
        for token in str(data.clean_text[i]).split():
            global_list.append(token)
          
    return(global_list)


'''
this function get global list we created above as an input and return dictionary with keys eaqual to unique words available at global list and corresponding values in integer form that shows the frequesncu of corresponding word in set of all reviews 
we can suppose this function sd  DF calculator for all words 
'''

def global_dictionary(global_list ):
    
    global_dict = {}
    words_counter = {}  # to calculate frequency of each word 
    
    for word in global_list: # moving on steemed version of  all words available in reviews 


        if word not in words_counter:
            words_counter[word] = 1 # if it is nor existed before put frequency equal to 1 
        else:
            words_counter[word] += 1  # if it existed also before increase frequency by 1


    for word in global_list:
        
        if word not in global_dict:
            global_dict[word] = words_counter[word]  # creat final goal dictionaty
   
    return(global_dict)



def quantile_finder(global_dict):
    
    main_list=[]
    important_dict ={}
    
    for key in global_dict:
        
        main_list.append(global_dict[key])
     
    upper_bound = np.quantile(main_list, .99) # create proper upper bound 
    lower_bound = np.quantile(main_list, .66) # create proper lower bound 

    return(lower_bound , upper_bound )

'''
This function gets as an input the dictionary of all stemmed vocabulary and lower bounds and upper bounds produced at previous 
function it applied quantile method and return dictionary. The dictionary has key values equal to vocabulary that are availabel 
cmong specified quantiles and values equal to number of times that vocabulary repeated so in this way we simply have dictionary 
that just keep most important and influencing words.
'''


def main_words(global_dict , lower_bound , upper_bound):
    
    important_dict={}
    
    for key in global_dict:
        
        if global_dict[key] < upper_bound and global_dict[key] > lower_bound: # choose ones among boundries
            if key not in important_dict:
                important_dict[key] = global_dict[key]
            
    return(important_dict)

'''
this function get one text and important_dictionary as an input and just keep words of text that are available 
in our dictionary and return list of those important words in text
'''
def important_text_process(text,important_dict):
    
    important_list = []
    
    for token in str(text).split():
        
        if token in important_dict:
            important_list.append(token)
        
                
    return(important_list)
'''
This function get dataset as an input and apply important_text_process to 'clean_text' columns of data and add extra columns
to the data that just keep importsnt words of steemed version of text 
'''
def important_column_data(data):
    
    data['Important_Words']=data['clean_text'].apply(lambda x : important_text_process(x ,important_dict))
        
    #print("Done.")
    

'''
this function id term frequency calculator and has the same structure same as one we had in previous assignments for specific unique product ID in merge list of all important words in last columns of our data and calculate term frequency of word and normalize it byt deviding to the length of final merged list 

'''
def tf_calculator(slice1,unique_products,num_product) :
    
    
    index=(np.where(slice1['ProductId'] == unique_products[int(num_product)]))[0].tolist()
    fara = {}
    merged_list_length = 0
    
    # merge last columns of data wich named Important_words and keep last version of processed and reduced reviews
    

    for i in range(len(index)):

        merged_text = slice1['Important_Words'][int(index[i])+1]

        for token in merged_text:
            
            if token not in fara: # counting term frequency of each word
                
                fara[token] = 1
            else:
                fara[token] += 1

        merged_list_length = merged_list_length + len(merged_text)  # calculate final length to use for normalizing 

    for key in fara:
        
            fara[key]=(fara[key]/(merged_list_length+1))  # normalization 
                
    return(fara)

'''
this function calculate the inverse data frequency among steemed reduced version of all reviews and and the structure of function is exaclly like what we had in previous assignments it merged all the reviws for each product id and counting IDF 
'''
def IDF_calculator(slice1,unique_product_ID) :
    
    DF={}
    IDF = {}
    for i in range(len(unique_product_ID)):
        
        index = (np.where(slice1['ProductId'] == str(unique_product_ID[i]) ) )[0].tolist()
        merged_list = []
        
        for i in index:
            
            merged_text =slice1['Important_Words'][i+1]
            
            for token in merged_text:
                
                merged_list.append(token)
                
        for word in np.unique(np.array(merged_list)):
            
            if word not in DF:
                
                DF[word] = 1
            else:
                DF[word] += 1
    
    for word in DF:
        if word not in IDF:
            
            IDF[word] = round(math.log(len(unique_product_ID) / DF[word]))
    
    return(IDF)
    

    
def get_relevant_words(components,features):
    
    components_features = {i: [] for i in range(len(components))} 
    n_comp = len(components)
    for i in range(n_comp):
        ith_comp = components[i]
        for coef,feat in zip(ith_comp,features):
            if  coef > 10**(-2):
                components_features[i].append(feat)

    relevant_words = list(set(list(components_features.values())[0]))

    return relevant_words


### adding columns of relevant eord to data set
def extracting_relevant(important_dictionary,relevant_words,num_product,unique_products):
    
    relevant_list_product=[]
   
    index=(np.where(slice1['ProductId'] == unique_products[int(num_product)]))[0].tolist()
    fara = {}
    merged_list_length = 0

    for i in range(len(index)):

        merged_text = slice1['Important_Words'][int(index[i])+1]
        
        
        for token in merged_text:
            if token in relevant_words :
                 relevant_list_product.append(token)

    return( relevant_list_product) 


'''
This function calculate clusters of given centroids and how it works is that at first step it calculate the distance of each available vectors with all centroids and then by chooosing the minimum distance recognize the centroid and cluster in which this vector belongs 

'''
def distance_calculator(vectors , centroids):
    
    clusters = {}
    for i in range(len(vectors)) : 
        distance = {}
        for center in centroids:
            
            a = [center , vectors[i]]
            ecled_distance = euclidean_distances(a,a)
            distance[ecled_distance[0][1]] = center
            s =[]
            
        for key , values in distance.items():
            
            s.append(key)
        final_distance = min(s)
        final_center = tuple(distance[final_distance])
        try :
            clusters[final_center].append(vectors[i] )
        except KeyError:
            clusters[final_center] = [vectors[i]]
            
            
    return(clusters)
'''
This function get centroids and clusters as an input and after calculating the mean of all clusters retuen these values as an 
new and updated centroids 
'''

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k],axis=0))
    return newmu

'''
This function calculate the variance between the vector of centroids 
'''
def variance_among_centroids(centroids):
    comparision = []

    comparision.append(np.var(centroids))
    
    return (comparision)

'''
calculating the variance between all vectors available in each cluster 
'''
def inner_variance(mu, clusters):
    inner_var = []
    keys = sorted(clusters.keys())
    for k in keys:
        inner_var.append(np.var(clusters[k]))
    return inner_var
'''
as long as centroids are not remain the same the convergence does not complete and so we will continue 
'''
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

'''
this function apply all above functions and return final centroids and clusters 
'''
def k_means(vectors,initial_centroids):
    
    clusters = distance_calculator(vectors , initial_centroids)
    new_centroids = reevaluate_centers(initial_centroids, clusters)
    while not has_converged(initial_centroids, new_centroids) :
        initial_centroids = new_centroids
        variance_centroids = variance_among_centroids(initial_centroids)
        clusters = distance_calculator(vectors , initial_centroids)
        inner_variance_centrs = inner_variance(initial_centroids, clusters)
        #inner_variance_centroids = np.mean(inner_variance_centrs)
        print('new cluster with variance equal to = ',variance_centroids)
        print('inner variance for clusters =',inner_variance_centrs)
        print('---------------------------------------------------------------------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------')
        new_centroids = reevaluate_centers(initial_centroids, clusters)

    
    return (new_centroids,clusters)

'''
this function specifies the cluster number for given product id 
'''

def cluster_specification(product_id,cluster_appender ,extra_product ):
    clusterNumber = 0
    if product_id in cluster_appender:
        
        clusterNumber = cluster_appender[product_id]
    else :
        
        try:
            cluster_num_index = extra_product.index(product_id)+1
            clusterNumber = extra_product[cluster_num_index]
        except:

            clusterNumber = 0
            
    
                             
    return(clusterNumber)

'''
Adding extra columns ad cluster number to mail data frame in order to do group by and other stuffd 
'''
def cluster_producer(data):
    
    data['cluster_number']=data['ProductId'].apply(lambda x : cluster_specification(x,cluster_appender ,extra_product ))
        
    print("Done.")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    