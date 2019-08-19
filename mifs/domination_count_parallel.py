# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# Import Necessary Libraries
import numpy as np
import time

# Define Fuction to Compute Domiation Count
def computeDominationCount(selected, left, FFMI, FCMI):
    
    
    tic = time.time()
    
    # Find Number of Selected Feature Till Now
    totalSelectedFeature = len(selected)
    #print("Selected Feature: ", selected)
    #print("Number of Selected Feature :", totalSelectedFeature)
    
    # Find Number of Left Feature Till Now
    totalLeftFeature = len(left)
    #print("left feature: ", left)
    #print("Number of Left Feature :", totalLeftFeature)
    
    # Filter out FFMI Matrix for Selected and Left Feature
    # Rows = Selected Features
    # Cols = Left Features
    selR = FFMI[selected, :]
    
    #print("Selected Rows: ", selR)
    selRC = selR[:, left]
    #print("Selected Row & Column: ", selRC)
    
    # Reshape Filtered Out FFMI Matrix
    selectedRowCols = selRC.reshape(totalSelectedFeature, totalLeftFeature)
    
    # Compute Average FFMI For Above Filterd out Matrix
    avgFFCorr = np.sum(selectedRowCols, axis = 0, keepdims = True) / totalSelectedFeature
    # print("Average FFMI of Left Feature:", avgFFCorr)
    
    # Initialize Domination Vector  and Dominated Vector 
    dominationVec = np.zeros((1, totalLeftFeature), dtype = int)
    dominatedVec = np.zeros((1, totalLeftFeature), dtype = int)

    # Compute Domination Count for Each Left Feature
    for i in range(0, totalLeftFeature-1):
        for j in range(i+1, totalLeftFeature):
            if(avgFFCorr[0, j] > avgFFCorr[0, i]):
                dominationVec[0, j] = dominationVec[0, j] + 1
            else:
                dominationVec[0, i] = dominationVec[0, i] + 1
     
    # FCMI For Left Feature  
    FCMI = FCMI.T[0, left]
    #print("FCMI of Left Feature: ", FCMI)
                
    # Compute Dominated Count
    for i in range(0, totalLeftFeature-1):
        for j in range(i+1, totalLeftFeature):
            if(FCMI[j] < FCMI[i]):
                dominatedVec[0, j] = dominatedVec[0, j] + 1
            else:
                dominatedVec[0, i] = dominatedVec[0, i] + 1
    
    tac = time.time()
    
    # Return Domination Count & Dominated Count
    return(dominationVec, dominatedVec, 1000 * (tac - tic))
