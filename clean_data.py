# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------

    ctg = CTG_features.drop(extra_feature, axis=1)
    c_ctg = ctg.to_dict(orient='series')

    for key in c_ctg.keys():
        c_ctg[key] = c_ctg[key].apply(pd.to_numeric, errors='coerce')
        c_ctg[key] = c_ctg[key].dropna()

    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_cdf = CTG_features.drop(extra_feature, axis=1)  # remove 'DR' feature
    c_cdf = c_cdf.to_dict(orient='series')

    for key in c_cdf.keys():  # for every column, seperately
        c_cdf[key] = pd.to_numeric(c_cdf[key], errors='coerce')  # replace non-numeric values with NaN
        probs = c_cdf[key].value_counts(normalize=True)  # calculate the probability of each element in the column
        new_vals = []
        for value in c_cdf[key]:
            if np.isnan(value):
                new_vals.append(np.random.choice(list(probs.keys()), p=list(probs.values)))  # if value=NaN -> replace it using the requested function
            else:
                new_vals.append(value)  # if value is numeric -> keep it
        c_cdf[key] = new_vals
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {}

    for key in c_feat.keys():
        stats = c_feat[key].describe()
        rel_stat = stats[3:]
        rel_stat = rel_stat.to_dict()
        d_summary[key] = rel_stat
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    tmp_dict = c_feat.to_dict(orient='series')
    for key in tmp_dict.keys():  # for every column, seperately
        no_outliers = []
        # statistical calculations
        Q1 = d_summary[key]['25%']
        Q3 = d_summary[key]['75%']
        IQR = Q3 - Q1
        low_val = Q1 - 1.5 * IQR
        high_val = Q3 + 1.5 * IQR
        for value in tmp_dict[key]:  # for every value in the column
            if value >= low_val and value <= high_val:
                no_outliers.append(value)
            else:
                no_outliers.append(np.nan)
        c_no_outlier[key] = no_outliers
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature = []
    feature_vals = c_cdf[feature]
    for val in feature_vals:
        if val <= thresh:
            filt_feature.append(val)
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    nsd_res = {}
    # Standardization and Normalization of the data
    for key in CTG_features.keys():  # for every column sepparately
        new_norm_vals = []
        mean = np.mean(CTG_features[key])  # the mean of each series
        std = np.std(CTG_features[key])  # the std of each series
        min = np.min(CTG_features[key])  # the min of each series
        max = np.max(CTG_features[key])  # the max of each series

        if mode == 'standard':
            for val in CTG_features[key]:  # normalize each value in the specific column
                norm_val = (val - mean) / std
                new_norm_vals.append(norm_val)
            nsd_res[key] = new_norm_vals

        elif mode == 'MinMax':
            for val in CTG_features[key]:  # normalize each value in the specific column
                norm_val = (val - min) / (max - min)
                new_norm_vals.append(norm_val)
            nsd_res[key] = new_norm_vals

        elif mode == 'mean':
            for val in CTG_features[key]:  # normalize each value in the specific column
                norm_val = (val - mean) / (max - min)
                new_norm_vals.append(norm_val)
            nsd_res[key] = new_norm_vals

        else:
            nsd_res[key] = CTG_features[key]

    # Statistical and graphical comparison of the normalized and original data
    if flag == True:

        plt.hist(nsd_res[x], bins=120)  # histograms of the scaled data
        plt.xlabel(x)
        plt.ylabel('Count')
        plt.title('Histogram of the scaled ' + x + ' data feature with selected mode = ' + mode)
        plt.show()

        plt.hist(nsd_res[y], bins=120)  # histograms of the scaled data
        plt.xlabel(y)
        plt.ylabel('Count')
        plt.title('Histogram of the scaled ' + y + ' data feature with selected mode = ' + mode)
        plt.show()

    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
