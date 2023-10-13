import numpy as np
import os.path
import matplotlib.pyplot as plt
import sys
import datetime
import matplotlib.cm as cm
import csv

import time

import sklearn
from scipy import stats
import sklearn.semi_supervised
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import argparse

import random
from sklearn.metrics import f1_score

#from matplotlib.axes._axes import _log as matplotlib_axes_logger


#matplotlib_axes_logger.setLevel('ERROR')


#########################

# number of multiprobe
num_multi = 1

# point size of phase diagram
p_size = 25

# initial number
initial_num = 10

# iteration number
itt_num = 100

# indipendent number
ind_num = 10

# method for multi
#0: only ranking, 1:combination, 2:distance, 3:look-ahead
multi_method = 0


#########################

random.seed(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--estimation", default='LP')
    parser.add_argument("--sampling", default='LC')
    args = parser.parse_args()

    input_data = args.input
    #input_data = 'training.csv'
    LP_algorithm = args.estimation #'LP', 'LS'                                                                   
    #LP_algorithm = 'LP'
    US_strategy = args.sampling  #'LC' ,'MS', 'EA', 'RS'
    #US_strategy = 'LC'
    is_output_img = True
    
    
    #----load data
    current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    f = open(os.path.join(current_dir, input_data), 'r')

    ternary_data_list = []
    
    label_list_true = []

    data_list = []

    data_list2d = []

    co_num=[]
    label_list_text = []
    reader = csv.reader(f)
    for row in reader:

        label_list_text.append(row[0])
        data_list.append([float(row[1]),float(row[2])])
        data_list2d.append([float(row[1]),float(row[2])])

    #################
    #################



    existed_phase=list(set(label_list_text))
    existed_phase.sort()


    for i in range(len(label_list_text)):

        for j in range(len(existed_phase)):

            if label_list_text[i]==existed_phase[j]:

                label_list_true.append(j)

        co_num.append(label_list_text[i].count("+")+1)


    total_index_list = [i for i in range(len(label_list_true))]



    f1_res_tot = []

    num_phase_res_tot = []
    
    num_data_res_tot = []

    max_label = len(existed_phase)
    color_list = [cm.rainbow(float(i)/(max_label)) for i in range(max_label+1)]


    for ind in range(ind_num):


        labeled_index_list = random.sample(total_index_list, initial_num)

        unlabeled_index_list = list(set(total_index_list) - set(labeled_index_list))

        f1_res = []

        num_phase_res = []

        num_data_res = []

        for itt in range(itt_num):


            print("*************************")
            print("independent run =", ind)
            print("iteration =", itt)

            num_training = len(labeled_index_list)

            print("number of training data =", num_training)


            label_list = []

            for i in range(len(label_list_true)):

                if i in labeled_index_list:
                    label_list.append(label_list_true[i])

                else:
                    label_list.append(-1)


            ###############


            label_train = np.copy(label_list)
            #label_index_list = range(len(data_list))
            #labeled_index_list = [i for i in range(num_training)]
            #unlabeled_index_list = list(set([i for i in range(len(data_list))]) - set(labeled_index_list))
            dimension = len(data_list[0])
    
            data_list = np.array(data_list)
            data_list2d = np.array(data_list2d)


            #max_label = np.max(list(set(label_list)))
            color_list = [cm.rainbow(float(i)/(max_label)) for i in range(max_label+1)]
          
            #random.shuffle(color_list)

            ss = StandardScaler()
            ss.fit(data_list)
            data_list_std = ss.transform(data_list)


            ###############

            def sampling(label_train,labeled_index_list,unlabeled_index_list):

                #----SAMPLING
                #label_train = np.copy(label_list)
                #label_train[unlabeled_index_list] = -1

                #estimate phase of each point
                if LP_algorithm == 'LS':
                    lp_model = sklearn.semi_supervised.LabelSpreading()
                elif LP_algorithm == 'LP':
                    lp_model = sklearn.semi_supervised.LabelPropagation()


                lp_model.fit(data_list_std, label_train)
                predicted_labels = lp_model.transduction_[unlabeled_index_list]
                predicted_all_labels = lp_model.transduction_
                label_distributions = lp_model.label_distributions_[unlabeled_index_list]
                label_distributions_all = lp_model.label_distributions_
                classes = lp_model.classes_


                #print(label_train, classes,  predicted_labels, predicted_all_labels, label_distributions)


                #calculate Uncertainly Score
                if US_strategy == 'E':
                    pred_entropies = stats.distributions.entropy(label_distributions.T)
                    u_score_list = pred_entropies/np.max(pred_entropies)
                        
                    # most uncertain point
                    uncertainty_index = [unlabeled_index_list[np.argmax(u_score_list)]]
                    US_point_prob = label_distributions[np.argmax(u_score_list)]

                    # all ranking of uncertain point
                    ranking = np.array(u_score_list).argsort()[::-1]
                    multi_uncertainty_index = [unlabeled_index_list[ranking[i]] for i in range(len(unlabeled_index_list))]


                elif US_strategy == 'LC':
                    u_score_list = 1 - np.max(label_distributions, axis = 1)

                    # most uncertain point
                    uncertainty_index = [unlabeled_index_list[np.argmax(u_score_list)]]
                    US_point_prob = label_distributions[np.argmax(u_score_list)]

                    # all ranking of uncertain point
                    ranking = np.array(u_score_list).argsort()[::-1]
                    multi_uncertainty_index = [unlabeled_index_list[ranking[i]] for i in range(len(unlabeled_index_list))]
                    #multi_uncertainty = [ternary_data_list[unlabeled_index_list[ranking[i]]] for i in range(len(unlabeled_index_list))]
                

                elif US_strategy == 'MS':

                    u_score_list = []
                    for pro_dist in label_distributions:
                        pro_ordered = np.sort(pro_dist)[::-1]
                        margin = pro_ordered[0] - pro_ordered[1]
                        u_score_list.append(margin)

                    u_score_list = 1-np.array(u_score_list)

                    # most uncertain point
                    uncertainty_index = [unlabeled_index_list[np.argmax(u_score_list)]]
                    US_point_prob = label_distributions[np.argmax(u_score_list)]

                    # all ranking of uncertain point
                    ranking = np.array(u_score_list).argsort()[::-1]
                    multi_uncertainty_index = [unlabeled_index_list[ranking[i]] for i in range(len(unlabeled_index_list))]
                    #multi_uncertainty = [ternary_data_list[unlabeled_index_list[ranking[i]]] for i in range(len(unlabeled_index_list))]



                elif US_strategy == 'RS':
                    
                    random_permutation = np.random.permutation(unlabeled_index_list)
                    u_score_list = [0.5 for i in range(len(label_distributions))]

                    # most uncertain point
                    uncertainty_index = [random_permutation[0]]
                    US_point_prob = label_distributions[np.argmax(u_score_list)]
                    
                    # all ranking of uncertain point
                    ranking = random_permutation
                    multi_uncertainty_index = list(random_permutation)


                dt_now = datetime.datetime.now()

            
                if output_rank == False:
                    print("Multiprobe by US score ranking")
                    for i in range(len(multi_uncertainty_index)):
                        print('Next Point Rank', i+1, ':', ternary_data_list[multi_uncertainty_index[i]],
                        u_score_list[ranking[i]],label_distributions[ranking[i]],multi_uncertainty_index[i])


                #print(len(multi_uncertainty_index))
                #print(len(u_score_list))
                #print(len(unlabeled_index_list))
                #print(label_distributions)
                #print(predicted_labels)

                
                return multi_uncertainty_index,label_distributions,u_score_list,ranking,predicted_all_labels
               
            
            output_rank = True
            multi_uncertainty_index,label_distributions,u_score_list,ranking,predicted_all_labels \
            = sampling(label_train,labeled_index_list,unlabeled_index_list)
            #output_rank = True


            ###############################
            # plot
            ###############################


            if is_output_img:
            
                fig_title = "Sampling by "+LP_algorithm+'+'+ US_strategy

                plt.rcParams['xtick.direction'] = 'in'
                plt.rcParams['ytick.direction'] = 'in'

                plt.figure(figsize=(12, 4.5))
                plt.rcParams["font.size"] = 13
                ax = plt.subplot(131)


                for i in unlabeled_index_list:
                    plt.scatter(data_list[i][0], data_list[i][1],  c='gray', marker = 's', s=p_size)
                for i in labeled_index_list:
                    plt.scatter(data_list[i][0], data_list[i][1],  c=color_list[label_list[i]], marker="o", s=p_size, label=existed_phase[label_list[i]])
                #plt.xlim([0,1])
                #plt.ylim([0,1])
                plt.title('Checked points', size = 16)



                fig = plt.subplot(132)

                u_score_colors = cm.Greens(u_score_list)
                for i in range(len(data_list[unlabeled_index_list])):
                    plt.scatter(data_list[unlabeled_index_list][i][0], data_list[unlabeled_index_list][i][1],  c= u_score_colors[i], marker = 's', s=p_size)
                #plt.xlim([0,1])
                #plt.ylim([0,1])
                plt.title('Uncertainty score', size = 16)


                ax = plt.subplot(133)

                #plt.xlim([0,1])
                #plt.ylim([0,1])
                for i in unlabeled_index_list:
                    plt.scatter(data_list[i][0], data_list[i][1],  c=color_list[predicted_all_labels[i]], marker = 's', s= p_size)

                pointed_label_list = []

                for i in labeled_index_list:

                    if label_list[i] not in pointed_label_list:

                        plt.scatter(data_list[i][0], data_list[i][1],  c=color_list[label_list[i]], marker="o", s=p_size, label = existed_phase[label_list[i]])

                        pointed_label_list.append(label_list[i])

                    else:
                        plt.scatter(data_list[i][0], data_list[i][1],  c=color_list[label_list[i]], marker="o", s= p_size)



                plt.title('Estimated', size = 16)
                #ax.legend(fontsize=5, loc='upper right')
                #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

                plt.suptitle(fig_title, fontsize=20)
                plt.tight_layout()
                plt.subplots_adjust(top=0.8)

                output_img_dir = 'data/'+LP_algorithm+'_'+US_strategy+'_'+str(ind).zfill(3)+'_'+str(itt).zfill(5)+'.png'
                plt.savefig(output_img_dir, dpi = 300)

                plt.close()


            ##################################
            #multiple proposals
            ##################################



            if multi_method == 0:

                ###############
                # only ranking
                ###############

                US_point = multi_uncertainty_index[0:num_multi]

                print(US_point)


            if multi_method == 1:

                ##############
                # combination
                ##############

                #hyperparameter
                combi_num = 3

                US_point = []
                combi = []

                for i in range(len(multi_uncertainty_index)):

                    if sorted(label_distributions[ranking[i]].argsort()[::-1][0:combi_num]) not in combi and len(US_point) < num_multi:
                        US_point.append(multi_uncertainty_index[i])
                        combi.append(sorted(label_distributions[ranking[i]].argsort()[::-1][0:combi_num]))

                print(US_point)


            if multi_method == 2:
                    
                ##############
                # distance
                ##############

                from scipy.spatial import distance

                #hyperparameter
                delta = 10

                US_point = []

                for i in range(len(multi_uncertainty_index)):

                    if i == 0: US_point.append(multi_uncertainty_index[i])

                    true_num = 0

                    for j in range(len(US_point)):

                        if distance.euclidean(ternary_data_list[US_point[j]], ternary_data_list[multi_uncertainty_index[i]]) > delta:
                            
                            true_num += 1

                    if true_num == len(US_point) and len(US_point) < num_multi: US_point.append(multi_uncertainty_index[i])


                print(US_point)


            if multi_method == 3:

                ###############################
                #multiprobe by look-ahead
                ###############################

                #hyperparameter
                see_phases = 3


                recommend_list = [multi_uncertainty_index[0]]
                recommend_prob = [1.0]

                US_point0 = multi_uncertainty_index[0]
                US_point_prob0 = np.copy(label_distributions[np.argmax(u_score_list)])

                for i in range(see_phases):
        
                    label_train_add1 = np.copy(label_train)
                    labeled_index_list_add1 = np.copy(labeled_index_list)

                    label_train_add1[US_point0] = US_point_prob0.argsort()[::-1][i]
                    prob1 = US_point_prob0[US_point_prob0.argsort()[::-1][i]]

                    labeled_index_list_add1 = np.append(labeled_index_list_add1, US_point0)
                    unlabeled_index_list_add1 = list(set([i for i in range(len(data_list))]) - set(labeled_index_list_add1))

                    multi_uncertainty_index1,label_distributions1,u_score_list,ranking,predicted_all_labels \
                    = sampling(label_train_add1,labeled_index_list_add1,unlabeled_index_list_add1)



                    recommend_list.append(multi_uncertainty_index1[0])
                    recommend_prob.append(prob1)

                    US_point1 = multi_uncertainty_index1[0]
                    US_point_prob1 = np.copy(label_distributions1[np.argmax(u_score_list)])


                    for j in range(see_phases):

                        label_train_add2 = np.copy(label_train_add1)
                        labeled_index_list_add2 = np.copy(labeled_index_list_add1)

                        label_train_add2[US_point1] = US_point_prob1.argsort()[::-1][j]
                        prob2 = US_point_prob1[US_point_prob1.argsort()[::-1][j]]

                        labeled_index_list_add2 = np.append(labeled_index_list_add2, US_point1)
                        unlabeled_index_list_add2 = list(set([i for i in range(len(data_list))]) - set(labeled_index_list_add2))

                        multi_uncertainty_index2,label_distributions2,u_score_list,ranking,predicted_all_labels \
                        = sampling(label_train_add2,labeled_index_list_add2,unlabeled_index_list_add2)
            
                        recommend_list.append(multi_uncertainty_index2[0])
                        recommend_prob.append(prob2*prob1)

                        US_point2 = multi_uncertainty_index2[0]
                        US_point_prob2 = np.copy(label_distributions2[np.argmax(u_score_list)])
                    

                        for k in range(see_phases):

                            label_train_add3 = np.copy(label_train_add2)
                            labeled_index_list_add3 = np.copy(labeled_index_list_add2)

                            label_train_add3[US_point2] = US_point_prob2.argsort()[::-1][k]
                            prob3 = US_point_prob2[US_point_prob2.argsort()[::-1][k]]

                            labeled_index_list_add3 = np.append(labeled_index_list_add3, US_point2)
                            unlabeled_index_list_add3 = list(set([i for i in range(len(data_list))]) - set(labeled_index_list_add3))

                            multi_uncertainty_index3,label_distributions3,u_score_list,ranking,predicted_all_labels \
                            = sampling(label_train_add3,labeled_index_list_add3,unlabeled_index_list_add3)
                
                            recommend_list.append(multi_uncertainty_index3[0])
                            recommend_prob.append(prob3*prob2*prob1)

     
            
                rank_prob = list(np.argsort(recommend_prob))[::-1]

                US_point = []

                for i in range(len(rank_prob)):

                    if recommend_list[rank_prob[i]] not in US_point:
                        US_point.append(recommend_list[rank_prob[i]])

                    if len(US_point) >= num_multi: break

                print(US_point)




            ####################################
            #evaluation
            ####################################


            print("f1 =", f1_score(label_list_true, list(predicted_all_labels), average="macro"))

            f1_res.append(f1_score(label_list_true, list(predicted_all_labels), average="macro"))

            label_list = []

            for i in range(len(label_list_true)):

                if i in labeled_index_list:
                    label_list.append(label_list_true[i])

            print("number of data =", len(labeled_index_list))
            print("number of phases =", len(set(label_list)))
        
            num_phase_res.append(len(set(label_list)))
            
            num_data_res.append(len(labeled_index_list))


            # add data            
            labeled_index_list = labeled_index_list + US_point

            unlabeled_index_list = list(set(unlabeled_index_list) - set(US_point))

            print(labeled_index_list)

        f1_res_tot.append(f1_res)

        num_phase_res_tot.append(num_phase_res)
        
        num_data_res_tot.append(num_data_res)


    f1_res_mean = np.mean(f1_res_tot, axis=0)
    f1_res_std = np.std(f1_res_tot, axis=0)

    num_phase_res_mean = np.mean(num_phase_res_tot, axis=0)
    num_phase_res_std = np.std(num_phase_res_tot, axis=0)

    num_data_res_mean = np.mean(num_data_res_tot, axis=0)
    num_data_res_std = np.std(num_data_res_tot, axis=0)

    f1=open('f1.txt', 'w')

    for i in range(len(f1_res_mean)):
        f1.write(str('%30.15f' % float(f1_res_mean[i])))
        f1.write('\t')
        f1.write(str('%30.15f' % float(f1_res_std[i])))
        f1.write('\t')
        f1.write(str('%30.15f' % float(num_data_res_mean[i])))
        f1.write('\n')
    
    f1=open('num_phase.txt', 'w')

    for i in range(len(num_phase_res_mean)):
        f1.write(str('%30.15f' % float(num_phase_res_mean[i])))
        f1.write('\t')
        f1.write(str('%30.15f' % float(num_phase_res_std[i])))
        f1.write('\t')
        f1.write(str('%30.15f' % float(num_data_res_mean[i])))
        f1.write('\n')
