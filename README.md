import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
%matplotlib inline
#read in data
motif_scores_df = pd.read_csv('/home/shs038/veh_kla/motif_scores_C57BL6J.tsv', sep='\t')
motif_scores_df.index = motif_scores_df['ID'].values
del motif_scores_df['ID']
#seperate data into veh only, kla only, and unchanged
veh_indices=motif_scores_df[motif_scores_df['Factors'].str.contains('atac_veh')].index.values
kla_incices=motif_scores_df[motif_scores_df['Factors'].str.contains('atac_kla')].index.values
veh_indices=set(veh_indices)
kla_incices=set(kla_incices)
veh_only=veh_indices-kla_incices
kla_only=kla_incices-veh_indices
unchanged=veh_indices.intersection(kla_incices)
veh_only=np.array(list(veh_only))
kla_only=np.array(list(kla_only))
unchanged=np.array(list(unchanged))
veh_only_df=motif_scores_df.loc[motif_scores_df.index.isin(veh_only)]
del veh_only_df['Factors']
del veh_only_df['chr']
kla_only_df=motif_scores_df.loc[motif_scores_df.index.isin(kla_only)]
del kla_only_df['Factors']
del kla_only_df['chr']
unchanged_df=motif_scores_df.loc[motif_scores_df.index.isin(unchanged)]
del unchanged_df['Factors']
del unchanged_df['chr']
#normalized by max 
max_veh=veh_only_df.max(axis=0)
max_kla=kla_only_df.max(axis=0)
max_unchanged=unchanged_df.max(axis=0)
veh_normalized_df=veh_only_df/max_veh
kla_normalized_df=kla_only_df/max_kla
unchanged_normalized_df=unchanged_df/max_unchanged
#remove negative value and NaN
veh_normalized_df[veh_normalized_df<0]=0
kla_normalized_df[kla_normalized_df<0]=0
unchanged_normalized_df[unchanged_normalized_df<0]=0
veh_normalized_plot=np.nan_to_num(veh_normalized_df)
kla_normalized_plot=np.nan_to_num(kla_normalized_df)
unchanged_normalized_plot=np.nan_to_num(unchanged_normalized_df)
#plot motif scores
sns.distplot(veh_normalized_plot[veh_normalized_plot!=0])
plt.ylabel('Frequency')
plt.xlabel('motif scores')
plt.title('veh')
plt.show()
sns.distplot(kla_normalized_plot[kla_normalized_plot!=0])
plt.ylabel('Frequency')
plt.xlabel('motif scores')
plt.title('kla')
plt.show()
sns.distplot(unchanged_normalized_plot[unchanged_normalized_plot!=0])
plt.ylabel('Frequency')
plt.xlabel('motif scores')
plt.title('unchanged')
#calculate correlation coefficient 
def find_correlation(df):
    '''
    input: a dataframe contains all motifs scores
    output: a dataframe contains correlation coefficient of each motif pai
    '''
    motifs = df.columns.values
    correlation_df=np.zeros((df.shape[1],df.shape[1]),dtype=np.float)
    correlation_df=pd.DataFrame(correlation_df, columns=motifs)
    correlation_df.index=motifs
    for i in range(df.shape[1]-1):
        for j in range(i+1,df.shape[1]):
            motif_paris=df.iloc[:,[i,j]]#select two moitfs
            #remove data that two motfis do not co-occur
            motif_paris=motif_paris[motif_paris.ix[:,0]!=0]
            motif_paris=motif_paris[motif_paris.ix[:,1]!=0]
            #calculate correlation
            coef=np.corrcoef(motif_paris.ix[:,0],motif_paris.ix[:,1])
            correlation_df.ix[i,j]=coef[0,1]
    #reshape dataframe
    Pairs=[]
    Correlation=[]
    #loop in part of count data that contain meaningful correlation
    for i in range (correlation_df.shape[1]-1):
        for j in range (i+1,correlation_df.shape[1]):
            #put motif pair and correlation into the empty list
            motif_pairs=(motifs[i],motifs[j])
            Pairs.append(motif_pairs)
            Correlation.append(correlation_df.ix[i,j])
    #reshape the dataframe
    reshaped_df=pd.DataFrame({'Correlation': Correlation}, index=Pairs)
    return reshaped_df
veh_correlation=find_correlation(veh_normalized_df)
kla_correlation=find_correlation(kla_normalized_df)
unchanged_correlation=find_correlation(unchanged_normalized_df)
#compare correlation coefficient of motif paris under veh and kla treatment
correlation_df=pd.concat([veh_correlation, kla_correlation,unchanged_correlation], axis=1)
correlation_df.columns = ['veh', 'kla','unchanged']
correlation_df=correlation_df.fillna(0)
#plot coefficient
sns.distplot(correlation_df['veh'])
plt.ylabel('Frequency')
plt.xlabel('Motif Correlation')
plt.title('veh') 
plt.show()
sns.distplot(correlation_df['kla'])
plt.ylabel('Frequency')
plt.xlabel('Motif Correlation')
plt.title('kla')
plt.show()
sns.distplot(correlation_df['unchanged'])
plt.ylabel('Frequency')
plt.xlabel('Motif Correlation')
plt.title('unchanged') 
correlation_df['veh-kla']=correlation_df['veh']-correlation_df['kla']
correlation_df['veh-unchanged']=correlation_df['veh']-correlation_df['unchanged']
correlation_df['kla-unchanged']=correlation_df['kla']-correlation_df['unchanged']
#plot difference
sns.distplot(correlation_df['veh-kla'])
plt.ylabel('Frequency')
plt.xlabel('veh-kla')
plt.title('Motif Correlation Difference') 
plt.show()
sns.distplot(correlation_df['veh-unchanged'])
plt.ylabel('Frequency')
plt.xlabel('veh-unchanged')
plt.title('Motif Correlation Difference under') 
plt.show()
sns.distplot(correlation_df['kla-unchanged'])
plt.ylabel('Frequency')
plt.xlabel('kla-unchanged')
plt.title('Motif Correlation Difference under') 
#Truth table of veh
veh_std=np.std(correlation_df['veh'])
veh_truth=1*(abs(correlation_df['veh'])>=2*veh_std)
veh_truth=veh_truth.to_frame(name=None)
#Truth table of kla
kla_std=np.std(correlation_df['kla'])
kla_truth=1*(abs(correlation_df['kla'])>=2*kla_std)
kla_truth=kla_truth.to_frame(name=None)
#Truth table of unchanged
unchanged_std=np.std(correlation_df['unchanged'])
unchanged_truth=1*(abs(correlation_df['unchanged'])>=2*unchanged_std)
unchanged_truth=unchanged_truth.to_frame(name=None)
#concatenate Truth able
Truth_table_scores=pd.concat([veh_truth, kla_truth,unchanged_truth], axis=1)
Truth_table_scores.to_csv('/home/shs038/veh_kla/Truth_table_scores.tsv', sep='\t')
