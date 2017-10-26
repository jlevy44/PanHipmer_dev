import begin
import subprocess
from collections import defaultdict
import scipy.sparse as sps
from scipy.stats import pearsonr
from itertools import combinations
import cPickle as pickle
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
# FIXME ADD DIMENSIONAL REDUCTION

def convertChr2ListIntervals(intervalDict,chunkSize):
    chunks = {key: (range(0,intervalDict[key],chunkSize) if intervalDict[key] % chunkSize == 0 else range(0,intervalDict[key],chunkSize) + [intervalDict[key]]) for key in intervalDict}
    return {key: [(chunks[key][i],chunks[key][i+1]) for i in range(len(chunks[key])-1)] for key in chunks}

@begin.subcommand
def eraseIndels(vcfIn,vcfOutName):
    subprocess.call("vcftools --vcf %s --remove-indels --recode --recode-INFO-all --out %s"%(vcfIn,vcfOutName),shell=True)

@begin.subcommand
def genSNPMat(vcfIn,samplingRate,chunkSize,test):
    try:
        samplingRate = int(samplingRate)
        test = int(test)
        chunkSize = int(chunkSize)
    except:
        samplingRate = 10000
        test = 0
        chunkSize = 100000
    chromosomes = defaultdict(list)
    encodeAlleles = {'0/0':0,'0/1':1,'1/1':2}
    colInfo = []
    rowInfo = []
    row, col, data = [],[],[]
    with open(vcfIn,'r') as f:
        for line in f:
            if line.startswith('##'):
                if line.startswith('##contig'):
                    lineInfo = map(lambda x: x.split('=')[1],line[line.find('<'):line.rfind('>')].split(','))
                    chromosomes[lineInfo[0]] = int(lineInfo[1])
            elif line.startswith('#CHROM'):
                colInfo = line.split()[9:]
            else:
                print f.tell()
                break
        print chromosomes, colInfo
        count = 0
        if test:
            for line in f: #FIXME does this remove the first line???
                if count % samplingRate == 0:
                    #print line
                    lineList = line.split()
                    rowInfo.append('_'.join(lineList[0:2]))
                    for idx,samp in filter(lambda y: y[1],enumerate(map(lambda x: encodeAlleles[x.split(':')[0]],lineList[9:]))):
                        row.append(len(rowInfo)-1)
                        col.append(idx)
                        data.append(samp)
                count += 1
                if count >= 10000:
                    break
        else:
            SNPs = defaultdict(list)
            SNPs_Used = []
            intervals = convertChr2ListIntervals(chromosomes,chunkSize)
            for line in f: #FIXME does this remove the first line???
                if count % samplingRate == 1:
                    offset = f.tell() # FIXME start here!?!?!?! also, start CNS run
                if count % samplingRate == 0:
                    lineList = line.split() # will this operation take too long??
                    SNPs[lineList[0]].append((int(lineList[1]),offset))
                count += 1
            for key in SNPs:
                SNPs[key] = np.array(SNPs[key])
            for chromosome in intervals: #FIXME Can I use BedTools to save time?
                for interval in intervals[chromosome]:
                    try:
                        correct_SNP = SNPs[chromosome][np.argmin(abs(SNPs[chromosome][:,0] - np.mean(interval)),axis=0),:]
                        rowInfo.append(chromosome+'-'+'_'.join(map(str,list(interval))))
                        f.seek(correct_SNP[1])
                        line = f.readline()
                        lineList = line.split()
                        SNPs_Used.append('_'.join(lineList[0:2]))
                        for idx,samp in filter(lambda y: y[1],enumerate(map(lambda x: encodeAlleles[x.split(':')[0]],lineList[9:]))):
                            row.append(len(rowInfo)-1)
                            col.append(idx)
                            data.append(samp)
                    except:
                        print chromosome, interval
            pickle.dump(SNPs_Used,open('used_SNPs.p','wb'),2)
    SNP_sample_dat = sps.coo_matrix((data, (row,col))).tocsr()
    print SNP_sample_dat
    rows_size = len(rowInfo)
    SNPMat = sps.dok_matrix((rows_size,rows_size))
    for i,j in combinations(range(rows_size),r=2):
        r = pearsonr(SNP_sample_dat.getrow(i).todense().T,SNP_sample_dat.getrow(j).todense().T)[0]
        SNPMat[i,j], SNPMat[j,i] = r, r
    SNPMat = SNPMat.tocsc()
    pickle.dump(colInfo,open('colNames.p','wb'),2)
    pickle.dump(rowInfo,open('rowNames.p','wb'),2)
    sps.save_npz('SNP_to_Sample.npz',SNP_sample_dat)
    sps.save_npz('SNP_to_SNP.npz',SNPMat)
    plt.figure()
    sns_plot = sns.heatmap(SNP_sample_dat.todense(),annot=False)
    plt.savefig('SNP_to_Sample.png')
    plt.figure()
    sns_plot = sns.heatmap(SNPMat.todense(),annot=False)
    plt.savefig('SNP_to_SNP.png')

@begin.start
def main():
    pass