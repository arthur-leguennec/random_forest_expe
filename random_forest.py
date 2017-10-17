# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:53:28 2017

@author: Arthur Le Guennec
"""

from laspy.file import File
from math import atan2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import itertools
import copy
import numpy as np
import scipy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

import os
import sys
import getopt
import argparse


def train_classifier(data, labels):
    labels = labels.reshape(-1, 1)
    random_forest = RandomForestClassifier(n_estimators = 100,
                                          criterion="gini",
                                          max_features="auto",
                                          oob_score=True,
                                          n_jobs=-1,
                                          verbose=1)
    

    random_forest.fit(data, labels)
    return random_forest


def test_classifier(model, data, labels = []):
    labels_predict = model.predict(data)
    confid_predict = model.predict_proba(data)
    confid_predict = np.max(confid_predict, axis=1)


    if (len(labels) == np.shape(data)[0]):
        labels = labels.reshape(-1)
        foo = np.equal(labels, labels_predict)
        error_rate = np.count_nonzero(foo) / len(labels)
        confid_good = confid_predict[foo]
        mean_confid = np.mean(confid_good)
        return labels_predict, confid_predict, error_rate, mean_confid
    else:
        return labels_predict, confid_predict


def load_las_file(name_dir_file):
    print('Loading file : ' + name_dir_file)
    inFile = File(name_dir_file, mode = "r")
    return inFile


def extract_features(inFile, name_features, scales):
    del name_features[0]
    del name_features[0]

    label = inFile.classification

    # Problème dans les données, certaines labels sont à 0
    ind = np.argwhere(label != 0)

    label = label[ind]

    #Modification des labels pour simplifier (ex: basse, moyenne et haute vegetation => classe vegetation)
    ind_tmp = np.argwhere(label == 3)
    label[ind_tmp] = 5  # low vegetation to high vegetation
    ind_tmp = np.argwhere(label == 4)
    label[ind_tmp] = 5  # medium to high vegetation
    ind_tmp = np.argwhere(label == 20)
    label[ind_tmp] = 5  # shrunk to high vegetation
    ind_tmp = np.argwhere(label == 7)
    label[ind_tmp] = 18  # low noise to high noise
    del ind_tmp
    

    column_names = [attr + '_s' + r.replace('.', '_') for r in scales for attr in name_features]

    column_names.insert(0, 'ratio_echo')

    features = None

    dat = inFile.get_num_returns()
    dat = inFile.get_num_returns()[ind].reshape(-1)
    features = dat

#    for dimension in inFile.point_format:
#        print(dimension.name)

    for name_feat in column_names:
        dat = inFile.reader.get_dimension(name_feat)[ind].reshape(-1)

        if (name_feat[0:16] == "diff_mean_height"):
            dat = np.nan_to_num(dat)
        else:
            dat = dat.copy()
            ind_nan = np.isnan(dat)
            dat[ind_nan] = -1

        if (features is None):
            features = dat
        else:
            features = np.vstack([features, dat])

    column_names.insert(0, 'num_returns')
    features = features.transpose()

    return features, label, column_names



def normalize_data(data, i=-1):
    if (i==-1):
        return

    return


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    cm = np.nan_to_num(cm)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def plot_feature_importance(feat_imp, column_names, indices=[],
                            title='Features importances'):
    """
    This function prints and plots the feautures importances.
    """

    if (indices == []):
        indices = range(len(feat_imp))

    plt.title(title)
    plt.bar(range(len(feat_imp)),
            feat_imp[indices],
            align="center",
            color="r")

    foo = column_names.copy()
    for i in range(len(indices)):
        column_names[i] = foo[indices[i]]
    column_names = tuple(column_names)

    plt.xticks(range(len(feat_imp)), column_names, rotation=90)
    plt.xlim([-1,len(feat_imp)])



def save_cloud(inFile, dir_out, filename, classif, conf):
    # Problème dans les données, certaines labels sont à 0
    label = inFile.classification
    ind = np.argwhere(label != 0)


    new_header = copy.copy(inFile.header)
    #Modification du point format id pour accéder au dimensions RGB
    new_header.data_format_id = 3
    #Create outfile
    outFile = dir_out + filename.replace('.laz', '_classify' + '.laz')
    outCloud = File(outFile, mode = "w", header=new_header, vlrs = inFile.header.vlrs)

    outCloud.define_new_dimension(name = 'classif',
                                  data_type = 9,
                                  description = "ntd")

    outCloud.define_new_dimension(name = 'conf',
                                  data_type = 9,
                                  description = "ntd")



    for dimension in inFile.point_format:
        if (dimension == inFile.point_format[13]):
            break
        dat = inFile.reader.get_dimension(dimension.name)[ind].reshape(-1)
#        dat = dat[ind]
        outCloud.writer.set_dimension(dimension.name, dat)


    multiscale_features_raw = pd.DataFrame()
    features_raw = np.hstack([classif.reshape(-1, 1), conf.reshape(-1, 1)])
    features_raw = pd.DataFrame(features_raw)
    features_raw.columns = np.array(["classif", "conf"])
    multiscale_features_raw = pd.concat([multiscale_features_raw, features_raw], axis=1)
    outCloud.classif = multiscale_features_raw.as_matrix(['classif']).ravel()
    outCloud.conf = multiscale_features_raw.as_matrix(['conf']).ravel()

    outCloud.close()


def existe(fname):
    try:
        f = open(fname,'r')
        f.close()
        return 1
    except:
        return 0









##################################################
##################################################
###            |\  /|   /\    |  |\  |         ###
##             | \/ |  /__\   |  | \ |          ##
###            |    | /    \  |  |  \|         ###
##################################################
##################################################
def main():
    dir_in = "../compute_descriptors/data_desc/"
    dir_out = "expe"

    parser = argparse.ArgumentParser()
    parser.add_argument('-dir_out')
    parser.add_argument('-scales', nargs='+')
    parser.add_argument('-feat_types', nargs='+')
    parser.add_argument('-test')

    args = parser.parse_args()

    if args.dir_out:
        dir_out = dir_out + args.dir_out + '/'
    else:
        dir_out = dir_out + '_nodirnames' + '/'

    if (not os.path.exists(dir_out)):    #Tu remplaces chemin par le chemin complet
        os.mkdir(dir_out)

    if args.test:
        dir_out = dir_out + args.dir_out + '/'
        feat_types = ["dim_c2", "dim_c2c3"]
        scales = ["0.25", "0.5", "1", "2", "2.5", "3", "4"]
        dir_out = dir_out + "test/"
        filenames_train_C2 = ["tile_C2_868500_6523000_ground_desc_s0_25_to_4.laz",
                              "tile_C2_869000_6523000_ground_desc_s0_25_to_4.laz",
                              "tile_C2_869500_6525000_ground_desc_s0_25_to_4.laz"]
    else:
        scales = args.scales
        print(scales)

        scales_attr = ""
        for r in scales:
            scales_attr = scales_attr + '_s' + r.replace('.', '_')

        print(args.feat_types)
        feat_types = args.feat_types
        for types in feat_types:
            dir_out = dir_out + types

        dir_out = dir_out + '_' + scales_attr + "/"
        print(dir_out)

        filenames_train_C2 = ["tile_C2_868000_6524000_ground_desc_s0_25_to_4.laz",
                              "tile_C2_868000_6524500_ground_desc_s0_25_to_4.laz",
                              "tile_C2_868500_6523000_ground_desc_s0_25_to_4.laz",
                              "tile_C2_868500_6524000_ground_desc_s0_25_to_4.laz",
                              "tile_C2_869000_6523000_ground_desc_s0_25_to_4.laz",
                              "tile_C2_869000_6524000_ground_desc_s0_25_to_4.laz",
                              "tile_C2_869000_6525000_ground_desc_s0_25_to_4.laz",
                              "tile_C2_869500_6524500_ground_desc_s0_25_to_4.laz",
                              "tile_C2_869500_6525000_ground_desc_s0_25_to_4.laz"]


#    feat_types = ["dim_c2", "dim_c2c3"]
#
#    scales = ["0.25", "0.5", "1", "2", "2.5", "3", "4"]
#    dir_out = dir_out + "_test/"
#


    if (not os.path.exists(dir_out)):    #Tu remplaces chemin par le chemin complet
        os.mkdir(dir_out)

    #scales = [0.25, 0.5, 1, 2, 2.5, 3, 4]

    class_names = ['Ground',
                   'Vegetation',
                   'Buildings',
                   'Water surface',
                   'Wires',
                   'Poles',
                   'Noise',
                   'Cars']

    class_labels = [2, 5, 6, 8, 13, 15, 18, 21]

    type_features = {}
    type_features["point_based"] = [0, 1]
    type_features["dim_c2"] = [2, 3, 4, 5]
    type_features["dim_c2c3"] = [6, 7, 8, 9]
    type_features["height_c2"] = [11]
    type_features["height_c2c3"] = [10, 11, 12]
    type_features["intensity_c2"] = [14]
    type_features["intensity_c2c3"] = [13, 14, 15]
    type_features["all_c2"] = [2, 3, 4, 5, 11, 14]
    type_features["all_c2c3"] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


    name_features = ['num_returns',
                     'echo_ratio',
                     'lambda1s_c2',
                     'lambda2s_c2',
                     'lambda3s_c2',
                     'slopes_c2',
                     'lambda1s_c2c3',
                     'lambda2s_c2c3',
                     'lambda3s_c2c3',
                     'slopes_c2c3',
                     'diff_mean_heights',
                     'stdh_c2',
                     'stdh_c3',
                     'rapport_mean_intensitys',
                     'std_intensitys_c2',
                     'std_intensitys_c3']

    features_choose = []
    for types in feat_types:
        features_choose = np.hstack([features_choose, type_features[types]])

    features_choose = np.int16(features_choose)

    foo = name_features.copy()
    name_features = []
    for i in range(len(features_choose)):
        name_features.append(foo[features_choose[i]])

    print(name_features)




    filename_valid_C2 = "tile_C2_869000_6523500_ground_desc_s0_25_to_4.laz"

    for filename in filenames_train_C2:
        inFile_C2 = load_las_file(dir_in + filename)
        if (filename == filenames_train_C2[0]):
            nam_feat = name_features.copy()

            [foo, labels_train, column_names] = extract_features(inFile_C2, nam_feat, scales)
            data_train = foo
        else:
            nam_feat = name_features.copy()
            [foo, lab_tmp, column_names] = extract_features(inFile_C2, nam_feat, scales)
            data_train = np.vstack([data_train, foo])
            #foo = inFile_C2.classification
            labels_train = np.concatenate([labels_train, lab_tmp])

    del foo

    inFile_C2 = load_las_file(dir_in + filename_valid_C2)

    nam_feat = name_features.copy()
    [data_valid, labels_valid, column_names] = extract_features(inFile_C2, nam_feat, scales)

    #labels_valid = inFile_C2.classification

    model_rf = train_classifier(data_train, labels_train)

    [labels_predict,
     confid_predict,
     error_rate,
     mean_confid] = test_classifier(model_rf, data_valid, labels_valid)

    conf_mat = confusion_matrix(labels_valid, labels_predict)
    print(error_rate)
    print(conf_mat)

    if existe("expe" + args.dir_out + "/result_error_rate_mean_conf.txt"):
        file_result = open("expe" + args.dir_out + "/result_error_rate_mean_conf.txt", "a")
    else:
        file_result = open("expe" + args.dir_out + "/result_error_rate_mean_conf.txt", "x")
    scales_attr = ""
    for r in scales:
        scales_attr = scales_attr + 's' + r
    file_result.write(scales_attr + ':\t' + str(error_rate) + '\t' + str(mean_confid) + '\n')
    file_result.close()

    i = 0
    for classe in class_names:
        if existe("expe" + args.dir_out + "/result_error_rate_mean_conf_" + classe + ".txt"):
            file_result = open("expe" + args.dir_out + "/result_error_rate_mean_conf_" + classe + ".txt", "a")
        else:
            file_result = open("expe" + args.dir_out + "/result_error_rate_mean_conf_" + classe + ".txt", "x")
        scales_attr = ""
        for r in scales:
            scales_attr = scales_attr + 's' + r

        ind_lab = np.argwhere(labels_valid == class_labels[i])

        [foo1,
         foo2,
         error_rate,
         mean_confid] = test_classifier(model_rf, data_valid[ind_lab], labels_valid[ind_lab])

        del foo1
        del foo2

        file_result.write(scales_attr + ':\t' + str(error_rate) + '\t' + str(mean_confid) + '\n')
        file_result.close()

        i = i + 1


    feat_imp = model_rf.feature_importances_


    plt.figure()
    plot_feature_importance(title="Features importances brut",
                            feat_imp=feat_imp,
                            indices=[],
                            column_names=column_names)

    plt.savefig(dir_out + 'fig_feat_imp_no_sorted.png')
    plt.savefig(dir_out + 'fig_feat_imp_no_sorted.svg')
    plt.close()

    indices = np.argsort(feat_imp)[::-1]


    plt.figure()
    plot_feature_importance(title="Features importances sorted",
                            feat_imp=feat_imp,
                            indices=indices,
                            column_names=column_names)


    plt.savefig(dir_out + 'fig_feat_imp_sorted.png')
    plt.savefig(dir_out + 'fig_feat_imp_sorted.svg')

    #plt.show()
    plt.close()

    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=True, classes=class_names,
                          title='Normalized confusion matrix')

    plt.savefig(dir_out + 'fig_conf_matrix_normalize.png')
    plt.savefig(dir_out + 'fig_conf_matrix_normalize.svg')

    #plt.show()
    plt.close()

    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False, classes=class_names,
                          title='Confusion matrix')

    plt.savefig(dir_out + 'fig_conf_matrix.png')
    plt.savefig(dir_out + 'fig_conf_matrix.svg')

    #plt.show()
    plt.close()

    save_cloud(inFile_C2, dir_out, filename_valid_C2, labels_predict, confid_predict)

    inFile_C2.close()

    print("Fin!")

    return

if __name__ == "__main__":
    main()






























#
