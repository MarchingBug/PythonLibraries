import logging

import azure.functions as func
import json

import scipy.stats # scientific python statistical package
from scipy.stats import  poisson,binom # binomial and poisson distribution functions
from array import *
import numpy as np
from collections import OrderedDict

def parse_json(total_nc,total_cc):
    logging.info("entering parse_json") 
    total_cc = list(total_cc)
    total_nc = list(total_nc)   
    
    x = { "total_cc":  [num  for num in total_cc], "total_nc": [num  for num in total_nc] }
    logging.info(x) 
    output = json.dumps(x)
    return output 



def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')    

    req_body = req.get_json()
    logging.info(req_body)
    x = req_body.get('x')
    TimeLag = req_body.get('T') 
    Per_loc = req_body.get('per_loc')
    Per_admit = req_body.get('per_admit')
    Per_cc = req_body.get('per_cc')
    logging.info(Per_cc)
    ILOS_cc = req_body.get('LOS_cc')
    
    ILOS_nc = req_body.get('LOS_nc')
    logging.info(ILOS_nc)
    newcases_json = req_body.get('newcases')      

    new_cases = json.dumps(newcases_json)
    new_cases = json.loads(new_cases)     
    new_cases = np.asarray(new_cases)
    
    
    logging.info(new_cases)

    t = int(TimeLag)    
    x = len(new_cases)   
    per_loc = float(Per_loc)     
    per_admit = float(Per_admit)
    per_cc = float(Per_cc)
    LOS_cc = int(ILOS_cc)
    LOS_nc = int(ILOS_nc)

    x_list = list(range(len(new_cases)))
    logging.info(x_list)   

    new_cases_lag = []
    for i in new_cases:      
       lag_pop  =  i* poisson.pmf(x_list, t)       
       new_cases_lag.append(lag_pop)           

    logging.info(new_cases_lag)

    lol = []
    for i, daily_vals in enumerate(new_cases_lag):
            # number of indices to pad in front
            fi = [0]*i
            diff = len(new_cases) - len(fi)
            # number of indices to pad in back
            bi = [0]*diff
            ls = list(fi) + list(daily_vals) + list(bi)
            lol.append(np.array(ls))
        
    # convert the list of time-staggered lists to an array
    ar = np.array(lol)
        
    # get the time-lagged sum of visits across days
    ts_lag = np.sum(ar, axis=0)
    # upper truncate for the number of days in observed y values
    ts_lag = ts_lag[:len(new_cases)]
    
    logging.info('Calculating beds')
    ########################## Calculate Bed Needs #################
    cc = per_cc * per_admit *  per_loc * np.array(ts_lag)
    cc = cc.tolist()
        
    nc = (1 - per_cc) * per_admit * per_loc * np.array(ts_lag)
    nc = nc.tolist()

    p = 0.5
    n_cc = LOS_cc*2
    n_nc = LOS_nc*2
       
    # get the binomial random variable properties
    rv_nc = binom(n_nc, p)
    # Use the binomial cumulative distribution function
    p_nc = rv_nc.cdf(np.array(range(1, len(x_list)+1)))
        
        # get the binomial random variable properties
    rv_cc = binom(n_cc, p)
        # Use the binomial cumulative distribution function
    p_cc = rv_cc.cdf(np.array(range(1, len(x_list)+1)))
        
    # Initiate lists to hold numbers of critical care and non-critical care patients
    # who are expected as new admits (index 0), as 1 day patients, 2 day patients, etc.
    LOScc = np.zeros(len(x_list))
    LOScc[0] = ts_lag[0] *  per_cc *  per_admit *  per_loc
    LOSnc = np.zeros(len(x_list))
    LOSnc[0] =  ts_lag[0] * (1- per_cc) *  per_admit *  per_loc
        
    total_nc = []
    total_cc = []
        
    # Roll up patient carry-over into lists of total critical care and total
    # non-critical patients expected
    for i, day in enumerate(x_list):
        LOScc = LOScc * (1 - p_cc)
        LOSnc = LOSnc * (1 - p_nc)
            
        LOScc = np.roll(LOScc, shift=1)
        LOSnc = np.roll(LOSnc, shift=1)

            
        LOScc[0] = ts_lag[i] *  per_cc *   per_admit * per_loc
        LOSnc[0] = ts_lag[i] * (1 -  per_cc) *  per_admit * per_loc
    
        total_nc.append(np.sum(LOSnc))
        total_cc.append(np.sum(LOScc))
    
    ############## Generate Json File #############################
        
    output = parse_json(total_nc,total_cc)
    
    ###############################################################


    
    #results = ts_lag.tolist()

    #logging.info(results)      

    return func.HttpResponse(
            output,
             status_code=200
        )
