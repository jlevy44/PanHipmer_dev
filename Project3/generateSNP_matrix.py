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
import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import plotly.figure_factory as ff
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix
from Bio import Phylo
import networkx as nx
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
    try:
        dimensions = int(dimensions)
    except:
        dimensions = 3
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
def build_distance_matrix(SNP_Samples_positions_npy, metric, output_fname, to_phylip_tree, samples_pickle, tree_method): #FIXME add output to format readable by phylip
    to_phylip_tree = int(to_phylip_tree)
    transformed_data = np.load(SNP_Samples_positions_npy)
    distance_matrix = pairwise_distances(transformed_data,metric=metric)
    np.save(output_fname, distance_matrix)
    if to_phylip_tree:
        if tree_method not in ['fitch','kitsch','neighbor']: #FIXME broken
            tree_method = 'neighbor'
        samples = pickle.load(open(samples_pickle,'rb'))
        constructor = DistanceTreeConstructor()
        dm = _DistanceMatrix(names=samples,matrix=[list(distance_matrix[i,0:i+1]) for i in range(len(samples))])
        tree = constructor.nj(dm)
        Phylo.write(tree,'output_tree_biopython.nh','newick')
        dm.format_phylip(open('distance_matrix.dat','w'))
        # FIXME need to output correct distance_matrix phylip file... There are errors
        # tree methods are fitch kitsch neighbor
        #with open('distance_matrix.dat','w') as f: # FIXME in future compute lower triangle to same memory .replace('.','').replace('-','').replace('l','')
        #    f.write(dm.format_phylip(open('distance_matrix.dat','w')))#f.write(str(len(samples)).ljust(10)+'\n'+str(dm))
            #f.write(str(len(samples)).ljust(10)+'\n'+'\n'.join(samples[i].replace('.','').replace('-','').ljust(10)+ '  ' + '  '.join(np.vectorize(lambda x: '%.4f'%x)(distance_matrix[i,:])) for i in range(len(samples))))
        #FNeighborCommandLine()
        #subprocess.call('f%s -datafile distance_matrix.dat'%tree_method,shell=True)


@begin.subcommand
def output_Dendrogram(distance_matrix_npy, samples_pickle):
    distance_matrix = np.load(distance_matrix_npy)
    dists = squareform(distance_matrix)
    linkage_mat = linkage(dists, 'single')
    samples = pickle.load(open(samples_pickle,'rb'))
    plt.figure()
    dendrogram(linkage_mat,labels=samples)
    plt.savefig('output_dendrogram.png')
    fig = ff.create_dendrogram(linkage_mat, orientation='left', labels=samples)
    fig['layout'].update({'width':1200, 'height':1800})
    py.plot(fig, filename='output_dendrogram.html')


@begin.subcommand
def reroot_Tree(tree_in,root_species,tree_out):
    t = Phylo.read(tree_in,'newick')
    t.root_with_outgroup(root_species)
    Phylo.write(t,tree_out,'newick')


@begin.subcommand
def build_Nearest_Neighbors(SNP_Samples_positions_npy, n_neighbors, metric, output_fname_npz):
    """Construct knearest neighbors graph """
    transformed_data = np.load(SNP_Samples_positions_npy)
    n_neighbors = int(n_neighbors)
    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm = 'brute' , metric=metric)
    neigh.fit(transformed_data)
    kneighbors = neigh.kneighbors_graph(transformed_data)
    sps.save_npz(output_fname_npz, kneighbors)

#FIXME add plotly function to plot graph using networkx and plotly, with ability to input initial positions or spectral etc

@begin.subcommand
def cluster_samples(samples_positions_npy, min_cluster_size, cluster_metric, alpha):
    min_cluster_size = int(min_cluster_size)
    alpha = float(alpha)
    transformed_data = np.load(samples_positions_npy)
    clusterer = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, metric = cluster_metric, alpha = alpha)
    clusters = clusterer.fit_predict(transformed_data)
    pickle.dump(clusters,open('clusters_output.p','wb'))
    plt.figure()
    clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    plt.savefig('cluster_Tree_Output.png') #FIXME add name of species

@begin.subcommand
def plotPositions(SNPs_or_Samples, positions_npy, labels_pickle, colors_pickle,output_fname, graph_file, layout, iterations):
    labels = pickle.load(open(labels_pickle,'rb'))
    iterations = int(iterations)
    if graph_file.endswith('.npz'):
        graph = 1
        G = nx.from_scipy_sparse_matrix(sps.load_npz(graph_file))
        mapping = {i:labels[i] for i in range(len(labels))}
        G=nx.relabel_nodes(G,mapping, copy=False)
        if layout == 'spectral':
            pos = nx.spring_layout(G,dim=3,iterations=iterations,pos=nx.spectral_layout(G,dim=3))
        elif layout == 'random':
            pos = nx.random_layout(G, dim=3)
        else:
            t_data = np.load(positions_npy)
            pos = nx.spring_layout(G,dim=3,iterations=iterations,pos={labels[i]: tuple(t_data[i,:]) for i in range(len(labels))})
        transformed_data = np.array([tuple(pos[k]) for k in labels]) # G.nodes()
        Xed = []
        Yed = []
        Zed = []
        for edge in G.edges():
            Xed += [pos[edge[0]][0], pos[edge[1]][0], None]
            Yed += [pos[edge[0]][1], pos[edge[1]][1], None]
            Zed += [pos[edge[0]][2], pos[edge[1]][2], None]
    else:
        transformed_data = np.load(positions_npy)
        graph = 0
    if output_fname.endswith('.html') == 0:
        output_fname += '.html'
    if colors_pickle.endswith('.p'):
        clusters = pickle.load(open(colors_pickle,'rb'))
        c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, int(np.max(clusters)) + 2)]
        names = np.vectorize(lambda x: 'Cluster %d'%x)(clusters)
        plots = []
        for cluster in set(clusters):
            plots.append(
                go.Scatter3d(x=transformed_data[clusters == cluster,0], y=transformed_data[clusters == cluster,1],
                             z=transformed_data[clusters == cluster,2],
                             name=names[clusters == cluster], mode='markers',
                             marker=dict(color=c[cluster], size=2), text=np.array(labels)[clusters == cluster]))
    else:
        N = 2
        c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N + 1)]
        plots = []
        plots.append(
            go.Scatter3d(x=transformed_data[:,0], y=transformed_data[:,1],
                         z=transformed_data[:,2],
                         name=SNPs_or_Samples, mode='markers',
                         marker=dict(color=c[0], size=2), text=labels))
    if graph:
        plots.append(go.Scatter3d(x=Xed,
                                  y=Yed,
                                  z=Zed,
                                  mode='lines',
                                  line=go.Line(color='rgb(210,210,210)', width=1),
                                  hoverinfo='none'
                                  ))
    fig = go.Figure(data=plots)
    py.plot(fig, filename=output_fname)




@begin.subcommand
def genSNPMat(vcfIn,samplingRate,chunkSize,test, SNP_op, grabAll, no_contig_info, encoded_vcf): # FIXME also add more vcf filters!! LD filter and ability to add reference species to data points (0,0,0) http://evomics.org/learning/population-and-speciation-genomics/fileformats-vcftools-plink/#ex3.2.2
    try:
        samplingRate = int(samplingRate)
        test = int(test)
        chunkSize = int(chunkSize)
        grabAll = int(grabAll)
        encoded_vcf = int(encoded_vcf)
        no_contig_info = int(no_contig_info)
    except:
        samplingRate = 10000
        test = 0
        chunkSize = 100000
        encoded_vcf = 0
        no_contig_info = 0
    chromosomes = defaultdict(list)
    encodeAlleles = {'0/0':0,'0/1':1,'1/1':2}
    if encoded_vcf: #FIXME how to find minor allele info
        encodeAlleles = {str(i):i for i in range(9)} # FIXME note that this only covers variants in one allele, not two, if no variant, 0, if 1st variant, 1, if 2nd variant, 2, if 3rd variant, 3, but 1, 2, 3 are uncorrelated!!! might as well make it binary
        encodeAlleles['.'] = -1 #FIXME change this encoding mapping, because N is not a lack of genetic variant
    colInfo = []
    rowInfo = []
    row, col, data = [],[],[]
    with open(vcfIn,'r') as f:
        for line in f:
            if line.startswith('##'):
                if line.startswith('##contig') and no_contig_info == 0:
                    lineInfo = map(lambda x: x.split('=')[1],line[line.find('<'):line.rfind('>')].split(','))
                    chromosomes[lineInfo[0]] = int(lineInfo[1])
            elif line.startswith('#CHROM'):
                colInfo = line.split()[9:]
                n_samples = len(colInfo)
            else:
                print f.tell()
                break
        #    offset = f.tell()
        #f.seek(offset) #FIXME Test!!
        print chromosomes, colInfo
        count = 0
        SNPs_Used = []
        if grabAll:
            for line in f: #FIXME does this remove the first line???
                #print line
                lineList = line.split()
                rowInfo.append('_'.join(lineList[0:2]))
                for idx,samp in filter(lambda y: y[1],enumerate(map(lambda x: encodeAlleles[x.split(':')[0]],lineList[9:]))):
                    if samp != 0:
                        row.append(len(rowInfo)-1)
                        col.append(idx)
                        data.append(samp)
            SNPs_Used = rowInfo
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
        else:
            SNPs = defaultdict(list)
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
            if no_contig_info:
                chromosomes = {chromosome: max(SNPs[chromosome][:,0]) for chromosome in SNPs}
            intervals = convertChr2ListIntervals(chromosomes,chunkSize)
            SNPLines = []
            for chromosome in intervals: #FIXME Can I use BedTools to save time?
                #print chromosome
                try:
                    snp_pos = SNPs[chromosome][:,0]
                    for interval in intervals[chromosome]:
                        try:
                            if SNP_op == 'positional_original':
                                correct_SNP = SNPs[chromosome][np.argmin(abs(snp_pos - np.mean(interval)),axis=0),:]
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
    if grabAll or test or SNP_op == 'positional_original':
        SNP_sample_dat = sps.coo_matrix((data, (row,col))).tocsr()
    else:
        #print SNPLines
        SNP_sample_dat = sps.vstack(SNPLines)
    print SNP_sample_dat
    pickle.dump(colInfo,open('samples.p','wb'),2)
    pickle.dump(rowInfo,open('regions.p','wb'),2)
    sps.save_npz('SNP_to_Sample.npz',SNP_sample_dat)
    if grabAll == 0:
        rows_size = len(rowInfo)
        SNPMat = sps.dok_matrix((rows_size,rows_size))
        #FIXME error here for true divide
        for i,j in combinations(range(rows_size),r=2):
            r = pearsonr(SNP_sample_dat.getrow(i).todense().T,SNP_sample_dat.getrow(j).todense().T)[0]
            SNPMat[i,j], SNPMat[j,i] = r, r
        for i in range(rows_size):
            SNPMat[i,i] = 1.
        SNPMat = SNPMat.tocsc()
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