import begin, sys
import subprocess
from collections import defaultdict
import scipy.sparse as sps
from scipy.stats import pearsonr
from itertools import combinations
import cPickle as pickle
import seaborn as sns
import numpy as np
import matplotlib
import scipy.signal as sg
from collections import Counter
matplotlib.use('Agg')
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
from scipy import stats
import plotly.graph_objs as go
import plotly.offline as py
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import FactorAnalysis, KernelPCA
# FIXME ADD DIMENSIONAL REDUCTION

def convertChr2ListIntervals(intervalDict,chunkSize):
    chunks = {key: (range(0,intervalDict[key],chunkSize) if intervalDict[key] % chunkSize == 0 else range(0,intervalDict[key],chunkSize) + [intervalDict[key]]) for key in intervalDict}
    return {key: [(chunks[key][i],chunks[key][i+1]) for i in range(len(chunks[key])-1)] for key in chunks}

@begin.subcommand
def eraseIndels(vcfIn,vcfOutName):
    subprocess.call("vcftools --vcf %s --remove-indels --recode --recode-INFO-all --out %s"%(vcfIn,vcfOutName),shell=True)

def snp_operation(f,lines,encodeAlleles,SNP_op,n_samples):
    #row, col, data = [],[],[]
    SNP_sample_dat = sps.dok_matrix((len(lines),n_samples))
    #print lines
    for count, line_num in enumerate(list(lines)):
        f.seek(line_num)
        lineList = f.readline().split()
        for idx,samp in filter(lambda y: y[1],enumerate(map(lambda x: encodeAlleles[x.split(':')[0]],lineList[9:]))):
            SNP_sample_dat[count,idx] = samp
            #row.append(count)
            #col.append(idx)
            #data.append(samp)
    #print row,col,data, 'b4'
    SNP_sample_dat = SNP_sample_dat.tocsc()#sps.coo_matrix((data, (row,col))).tocsc()
    print SNP_sample_dat.toarray()
    if SNP_op == 'mode':
        return sps.csr_matrix(stats.mode(SNP_sample_dat.todense(),axis=0)[0]) #FIXME start here
    elif SNP_op == 'mean':
        return sps.csr_matrix(SNP_sample_dat.mean(axis=0))
    elif 'smooth' in SNP_op:
        SNP_sample_dat = SNP_sample_dat.toarray()
        smooth = np.hstack([ sg.savgol_filter(y.T,17,13).T for y in np.hsplit(SNP_sample_dat,np.shape(SNP_sample_dat)[1])])
        if SNP_op == 'smooth_avg':
            return sps.csr_matrix(smooth.mean(axis=0))
        elif SNP_op == 'smooth_positional':
            return sps.csr_matrix(smooth[int(np.shape(smooth)[1]/2.),:])
        else:
            return SNP_sample_dat[int(len(SNP_sample_dat)/2.),:]
    elif 'poly' in SNP_op:
        SNP_sample_dat = SNP_sample_dat.toarray() # FIXME spit out values below and fit function... No KDE??
        counts = [np.array([(i,val) for i,val in enumerate(Counter(y).values())]) for y in np.hsplit(SNP_sample_dat,np.shape(SNP_sample_dat)[1])]
        polyfnc = [np.vectorize(lambda x: np.polyfit(count[:,0],count[:,1],4)*np.array([x**4,x**3,x**2,x,1]).T)(np.arange(0,2.01,.01)) for count in counts]
        if SNP_op == 'poly_avg':
            return sps.csr_matrix(np.array([np.mean(polyfnc[i]) for i in range(len(polyfnc))]))
        elif SNP_op == 'max_poly':
            return sps.csr_matrix(np.array([np.max(polyfnc[i]) for i in range(len(polyfnc))]))
        else:
            return SNP_sample_dat[int(len(SNP_sample_dat)/2.),:]
    else:
        return SNP_sample_dat[int(len(SNP_sample_dat)/2.),:]

@begin.subcommand
def dimReduce(matrix_file, samples_or_SNPs, reduction_technique, dimensions, transform_metric, out_fname):
    dimensions = int(dimensions)
    if dimensions > 3 or dimensions < 3:
        print 'Reduction will complete, but will be unable to plot results in 3D.'
    matrix_SNP = sps.load_npz(matrix_file)
    dimensionalityReducers = {'kpca': KernelPCA(n_components=dimensions,kernel=transform_metric), 'factor': FactorAnalysis(n_components=dimensions),
                                  'feature': FeatureAgglomeration(n_clusters=dimensions)}
    if samples_or_SNPs == 'samples':
        matrix_SNP = matrix_SNP.T
    if reduction_technique != 'kpca':
        matrix_SNP = matrix_SNP.toarray()
    transformed_data = dimensionalityReducers[reduction_technique].fit_transform(matrix_SNP)
    np.save(out_fname,transformed_data)


@begin.subcommand
def plotPositions(SNPs_or_Scaffolds,positions_npy,labels_pickle,output_fname):
    transformed_data = np.load(positions_npy)
    labels = pickle.load(open(labels_pickle,'rb'))
    if output_fname.endswith('.html') == 0:
        output_fname += '.html'
    N = 2
    c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N + 1)]
    plots = []
    plots.append(
        go.Scatter3d(x=transformed_data[:,0], y=transformed_data[:,1],
                     z=transformed_data[:,2],
                     name=SNPs_or_Scaffolds, mode='markers',
                     marker=dict(color=c[0], size=2), text=labels))
    fig = go.Figure(data=plots)
    py.plot(fig, filename=output_fname)




@begin.subcommand
def genSNPMat(vcfIn,samplingRate,chunkSize,test, SNP_op, grabAll):
    try:
        samplingRate = int(samplingRate)
        test = int(test)
        chunkSize = int(chunkSize)
        grabAll = int(grabAll)
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
                n_samples = len(colInfo)
            else:
                print f.tell()
                break
        print chromosomes, colInfo
        count = 0
        SNPs_Used = []
        if grabAll:
            for line in f: #FIXME does this remove the first line???
                #print line
                lineList = line.split()
                SNPs_Used.append('_'.join(lineList[0:2]))
                rowInfo.append('_'.join(lineList[0:2]))
                for idx,samp in filter(lambda y: y[1],enumerate(map(lambda x: encodeAlleles[x.split(':')[0]],lineList[9:]))):
                    if samp != 0:
                        row.append(len(rowInfo)-1)
                        col.append(idx)
                        data.append(samp)
            pickle.dump(SNPs_Used,open('used_SNPs.p'),protocol=2)
        elif test:
            for line in f: #FIXME does this remove the first line???
                if count % samplingRate == 0:
                    #print line
                    lineList = line.split()
                    SNPs_Used.append('_'.join(lineList[0:2]))
                    rowInfo.append('_'.join(lineList[0:2]))
                    for idx,samp in filter(lambda y: y[1],enumerate(map(lambda x: encodeAlleles[x.split(':')[0]],lineList[9:]))):
                        row.append(len(rowInfo)-1)
                        col.append(idx)
                        data.append(samp)
                count += 1
                if count >= 10000:
                    break
            pickle.dump(SNPs_Used,open('used_SNPs.p'),protocol=2)
        else:
            SNPs = defaultdict(list)
            intervals = convertChr2ListIntervals(chromosomes,chunkSize)
            for line in f: #FIXME does this remove the first line???
                #if count % samplingRate == 1:
                #    offset = f.tell() # FIXME start here!?!?!?! also, start CNS run HELP!!!
                if count % samplingRate == 0:
                    offset = f.tell()
                    lineList = line.split() # will this operation take too long??
                    SNPs[lineList[0]].append((int(lineList[1]),offset))
                    if SNP_op != 'positional_original':
                        SNPs_Used.append('_'.join(lineList[0:2]))
                count += 1
            for key in SNPs:
                SNPs[key] = np.array(SNPs[key])
            #print SNPs
            SNPLines = []
            for chromosome in intervals: #FIXME Can I use BedTools to save time?
                #print chromosome
                try:
                    snp_pos = SNPs[chromosome][:,0]
                    for interval in intervals[chromosome]:
                        try:
                            if SNP_op == 'positional_original':
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
                            else:
                                print interval, snp_pos
                                lines = SNPs[chromosome][(snp_pos<interval[1])&(snp_pos>=interval[0]),1]
                                if list(lines):
                                    rowInfo.append(chromosome+'-'+'_'.join(map(str,list(interval))))
                                    SNPLines.append(snp_operation(f,lines,encodeAlleles,SNP_op,n_samples))
                                #SNPs_Used.append(SNPs_Used_new)
                        except:
                            pass
                except:
                    print chromosome
            pickle.dump(SNPs_Used,open('used_SNPs.p','wb'),2)
    if SNP_op == 'positional_original':
        SNP_sample_dat = sps.coo_matrix((data, (row,col))).tocsr()
    else:
        #print SNPLines
        SNP_sample_dat = sps.vstack(SNPLines)
    print SNP_sample_dat
    rows_size = len(rowInfo)
    SNPMat = sps.dok_matrix((rows_size,rows_size))
    for i,j in combinations(range(rows_size),r=2):
        r = pearsonr(SNP_sample_dat.getrow(i).todense().T,SNP_sample_dat.getrow(j).todense().T)[0]
        SNPMat[i,j], SNPMat[j,i] = r, r
    for i in range(rows_size):
        SNPMat[i,i] = 1.
    SNPMat = SNPMat.tocsc()
    pickle.dump(colInfo,open('samples.p','wb'),2)
    pickle.dump(rowInfo,open('regions.p','wb'),2)
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