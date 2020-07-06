'''pca_svd.py
Subclass of PCA_COV that performs PCA using the singular value decomposition (SVD)
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np

import pca_cov


class PCA_SVD(pca_cov.PCA_COV):
    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars` using SVD

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        TODO:
        - This method should mirror that in pca_cov.py (same instance variables variables need to
        be computed).
        - There should NOT be any covariance matrix calculation here!
        - You may use np.linalg.svd to perform the singular value decomposition.
        '''

        self.vars = vars
        rel_data = self.data.loc[:, vars]
        rel_data = np.array(rel_data)

        self.orig_scales = np.max(rel_data, axis = 0) - np.min(rel_data, axis = 0)
        self.orig_means = np.mean(rel_data, axis = 0)

        if normalize:
            rel_data = (rel_data - np.min(rel_data, axis = 0))/(np.max(rel_data, axis = 0)-np.min(rel_data, axis = 0))
            self.normalized = True
        else:
            self.normalized = False
        
        rel_data = np.array(rel_data)
        self.A = rel_data
        
        N = self.A.shape[0]
        Ac = self.A - np.mean(self.A, axis = 0)

        U, S, vT = np.linalg.svd(Ac)
        S2 = S * S
        self.e_vals = S2/(N-1)
        self.e_vecs = vT.T

        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)


        pass
