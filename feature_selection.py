import numpy as np
from scipy.linalg import qr
import copy
from scipy.stats import pearsonr, spearmanr
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def use_n_most_correlated(x,y, feature_names, n=5, cutoff=0.85, x_others=[]):
    #this function returns the n features that are the most correlated with the target variable (importance). Furthermore, they are first put through a 
    #pairewise correlation filter to ensure that they are not themselves more than 80% correlated
    feature_names = copy.deepcopy(feature_names)
    x_others = copy.deepcopy(x_others)
    x_pw, feature_names_pw, deleted_features_pw, x_others_pw = pairwise_correlation_filter(x, feature_names, x_others=x_others, cutoff=cutoff)

    #now sort these features by their correlation with y
    x_sorted, feature_names_sorted, x_others_sorted = sort_by_correlation(x_pw, y, feature_names_pw, corr_type="spearman", x_others=x_others_pw)


    for idx in range(len(x_others_sorted)):
        x_others_sorted[idx] = x_others_sorted[idx][:,0:n]
    x_n = x_sorted[:,0:n]
    feature_names_n = feature_names_sorted[0:n]
    #now want to return the top n features (first in sorted list)
    return x_n, feature_names_n, deleted_features_pw, x_others_sorted

def sort_by_correlation(x, y, feature_names, corr_type="spearman", x_others=[]):
    #take a design matrix and sort the columns according to their correlation with the target variable.
    x_others = copy.deepcopy(x_others)
    x = copy.deepcopy(x)

    correlations = []
    new_names = []
    for col in range(x.shape[1]):
        if corr_type == "pearson":
            corr, _ = pearsonr(x[:, col], y)
        elif corr_type == "spearman":
            corr, _ = spearmanr(x[:, col], y)
        correlations.append([col,corr])
    correlations = sorted(correlations, key=lambda x: np.abs(x[1]), reverse=True )    
    new_x = np.zeros(x.shape)
    new_x_others = []
    for x_other in x_others:
        new_x_others.append(np.zeros(x_other.shape))

    for v, val in enumerate(correlations):
        new_x[:,v] = x[:, val[0]]   
        new_names.append(feature_names[val[0]])
        for x_idx, x_other in enumerate(new_x_others):
            new_x_others[x_idx][:,v] = x_others[x_idx][:,val[0]]

    # #make sure there aren't more features than data points
    if new_x.shape[1] > 800:
        num_features = 800
        new_x = new_x[:,0:num_features]    

        for n, new_x_other in enumerate(new_x_others):
            new_x_others[n] = new_x_other[:,0:num_features]    
        new_names = new_names[0:num_features]  

    return new_x, new_names, new_x_others 


def lincom(x, feature_names, cutoff=0.05, deleted_features=None, x_others=[], scale_data=False):
    #take design matrix x and extract features using linear combinations filter (QR decomposition)

    x = copy.deepcopy(x)
    x_others = copy.deepcopy(x_others)
    if scale_data == True:    #scales when function first called , not for recursive calls
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        
        for idx, x_other in enumerate(x_others):
            x_others[idx] = scaler.transform(x_other)

    feature_names = copy.deepcopy(feature_names)
    q,r = qr(x, mode='economic')
    col_count = x.shape[1]
    abs_diag = np.abs(np.diag(r))

    #get the absolute value of the original columns as well
    abs_cols = np.zeros(col_count)
    for col in range(col_count):
        a = x[:,col]


        col_abs = np.sqrt(np.sum(a**2))
        abs_cols[col] = col_abs

    #tol = dim_max * dim_max * np.finfo(float).eps
    col_removed = False

    #get ratio of r diagonal elements to corresponding colum (fraction of magnitude after removing all projections)  
    r_col_ratios = abs_diag / abs_cols  
    for i in reversed(range(col_count)):
        if r_col_ratios[i] <= cutoff:    #fraction of vector after projections removed is less than cutoff
            #remove the column
            x = np.delete(x, np.s_[i], axis=1)
            col_removed = True

            if feature_names != None:
                feature_names.pop(i)
            for x_idx, x_other in enumerate(x_others):
                x_others[x_idx] = np.delete(x_other, np.s_[i], axis=1)    
            break


    if col_removed:
        x, feature_names, deleted_features, x_others = lincom(x, feature_names=feature_names, cutoff=cutoff, deleted_features=deleted_features, x_others=x_others, scale_data=False)    #iterate again until no columns removed

    #now don't allow more than 10 features to be returned
    if x.shape[1] > 10:
        x = x[:,:10]
        feature_names = feature_names[0:5]
        for idx, arr in enumerate(x_others):
            x_others[idx] = arr[:,:5]
    return x, feature_names, deleted_features, x_others
            
    #delete dropped features from features array 

def pairwise_correlation_filter(x, feature_names, deleted_features=None, cutoff=0.95, x_others=[]):
    #takes a design matrix, and remove features until there are none with a pairwise correlation greater than the cutoff. First look at two predictors with largest corr
    #and remove one with highest avg corr. with other features.
    feature_names = copy.deepcopy(feature_names)
    x_others = copy.deepcopy(x_others)
    x = copy.deepcopy(x)

    num_features = x.shape[1]
    #corr_matrix = np.zeros([num_features, num_features])
    # for r_1 in range(num_features):
    #     for r_2 in range(r_1+1, num_features):
                
    #         corr, _ = spearmanr(x[:,r_1], x[:, r_2])
    #         if math.isnan(corr):
    #             corr_matrix[r_1,r_2] = 0
    #             corr_matrix[r_2,r_1] = 0
    #         else:    
    #             corr_matrix[r_1,r_2] = np.abs(corr)
    #             corr_matrix[r_2,r_1] = np.abs(corr)

    #     corr_matrix[r_1, r_1] = 0 #to prevent self correlation from being the max        

    #now remove features until all correlations are less than the cutoff
    corr_matrix, _ = spearmanr(x, axis=0)
    corr_matrix = np.nan_to_num(corr_matrix)
    corr_matrix = np.abs(corr_matrix)
    np.fill_diagonal(corr_matrix, 0)

    r_1_max, r_2_max = np.unravel_index(np.argmax(corr_matrix, axis=None), corr_matrix.shape)
    max = corr_matrix[r_1_max, r_2_max]
    while max > cutoff:

        #delete the one with the highest avg corr with others. 
        avg_1 = np.average(corr_matrix[r_1_max,:])    
        avg_2 = np.average(corr_matrix[r_2_max,:])  
        if avg_1 > avg_2:
            x = np.delete(x, r_1_max, axis=1)
            corr_matrix = np.delete(corr_matrix, r_1_max, axis=0)
            corr_matrix = np.delete(corr_matrix, r_1_max, axis=1)
            feature_names.pop(r_1_max)
            for x_idx, x_other in enumerate(x_others):
                x_others[x_idx] = np.delete(x_other, r_1_max, axis=1)
            
        else:
            x = np.delete(x, r_2_max, axis=1)    
            corr_matrix = np.delete(corr_matrix, r_2_max, axis=0)
            corr_matrix = np.delete(corr_matrix, r_2_max, axis=1)
            feature_names.pop(r_2_max)
            for x_idx, x_other in enumerate(x_others):
                x_others[x_idx] = np.delete(x_other, r_2_max, axis=1)
        
        r_1_max, r_2_max = np.unravel_index(np.argmax(corr_matrix, axis=None), corr_matrix.shape)
        max = corr_matrix[r_1_max, r_2_max] 
   
    return x, feature_names, deleted_features, x_others

def pca(x, feature_names, deleted_features=None, cutoff=5, x_others=[], scale_data=True, print_top_features=False):
    if scale_data == True:    #scales when function first called , not for recursive calls
        scaler = StandardScaler()
        x = copy.deepcopy(x)
        scaler.fit(x)
        x = scaler.transform(x)
        x_others = copy.deepcopy(x_others)
        for idx, x_other in enumerate(x_others):
            x_others[idx] = scaler.transform(x_other)

    new_features = []    
    #compute svd
    U, S, VT = np.linalg.svd(x, full_matrices=0)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax1.semilogy(S, '-o', color='k')
    # ax2 = fig.add_subplot(122)
    # ax2.plot(np.cumsum(S)/np.sum(S), '-o', color='k')
    # plt.show()
    
    #we want to project the data onto only the first n principal components such that at least 0.9 of the variance is captured.
    #first get how many we should keep.

    #variance_captured = np.cumsum(S)/np.sum(S)
    n_keep = cutoff#max(np.argmax(variance_captured > cutoff) , 1)    #gets first occurrence greater than cutoff (max 1 for if all are above cutoff)
    #now need a new design matrix with n columns
    new_x = np.zeros([x.shape[0], n_keep])
    for j in range(n_keep):
        new_features.append(str("mixed_feature_" + str(j)))

    #now need to project features onto each of these new principal components and add as a new column
    for i in range(x.shape[0]):
        for j in range(n_keep):
            projected_feature = VT[j,:] @ x[i,:].T
            new_x[i,j] = projected_feature
    new_x_others = []
    for x_other in x_others:
        new_x_others.append(np.zeros([x_other.shape[0], n_keep]))
        for i in range(x_other.shape[0]):
            for j in range(n_keep):
                projected_feature = VT[j,:] @ x_other[i,:].T
                new_x_others[-1][i,j] = projected_feature
    feature_importances = np.zeros((800))      
    for j in range(n_keep):
        pc = VT[j,:]*S[j]
        feature_importances += pc
    #now need to print the top 20 important ones:
    if print_top_features == True:
        feature_importances = (feature_importances- np.amin(feature_importances))/ (np.amax(feature_importances)-np.amin(feature_importances))
        important_feature_inds = np.argsort(feature_importances) [::-1]   
        for i in range(21):
            print(f"{feature_names[important_feature_inds[i]]}: {feature_importances[important_feature_inds[i]]}")
                
    return new_x, new_features, deleted_features, new_x_others



if __name__ == "__main__":
    x = np.random.rand(400,200)

    x_new, _ = lincom(copy.deepcopy(x), cutoff=0.3)
    print("Done")