'''
Created on Jan 10, 2015

@author: zhangfaen
'''

import numpy as np

def pca(dataMat, moveToMeans, topNfeat=999999):
    print "dataMat:"
    print dataMat
    meanVals = np.mean(dataMat, axis=0)
    print "meanVals:"
    print meanVals
    if moveToMeans:
        meanRemoved = dataMat - meanVals
    else:
        meanRemoved = dataMat
    print "meanRemoved:"
    print meanRemoved
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    print "eigVals, eigVects"
    print eigVals
    print eigVects
    eigValInd = np.argsort(eigVals)
    print "eigValInd:"
    print eigValInd
    eigValInd = eigValInd[: -(topNfeat + 1) : -1]
    print "eigValInd trunked:"
    print eigValInd
    redEigVects = eigVects[:, eigValInd]
    print "redEigVects trunked"
    print redEigVects
    lowDDataMat = meanRemoved * redEigVects
    print "lowDDataMat:"
    print lowDDataMat

def svd(dataMat, moveToMeans):
    meanVals = np.mean(dataMat, axis=0)
    print "meanVals:"
    print meanVals
    if moveToMeans:
        meanRemoved = dataMat - meanVals
    else:
        meanRemoved = dataMat
    u, s, v = np.linalg.svd(meanRemoved)
    print u
    print s
    print v
if __name__ == '__main__':
    # (1, 1)
    # (1,5, 1.1)
    # (2, 1)
    dataMat = np.mat([[1, 1], [1.5, 1.1], [2, 1]])
    pca(dataMat, True, 1)
    print "--------------------------------------"
    pca(dataMat, False, 1)
    print "--------------------------------------"
    svd(dataMat, True)
    print "--------------------------------------"
    svd(dataMat, False)
    