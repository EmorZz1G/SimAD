from .f1_score_f1_pa import *
from .fc_score import *
from .precision_at_k import *
from .customizable_f1_score import *
from .AUC import *
from .Matthews_correlation_coefficient import *
from .affiliation.generics import convert_vector_to_events
from .affiliation.metrics import pr_from_events
from .vus.metrics import get_range_vus_roc

def test_bias():
    b = get_bias("UCR_AUG",237,aff=1)
    # print(b)
    b2 = get_bias("MSL",237,aff=1)
    pass
    print(b,b2)

others_pre_bias = """0.105410468
0.121298616
0.058084064
0.127917261
0.275074599
0.325183106
0.012054811"""

others_aff_pre_bias="""0.513357293
0.529361619
0.54805711
0.514772486
0.531669087
0.548291256
0.502068058"""

bias_name="""MSL
SWAT
WADI
SMAP
PSM
NIPS_TS_Swan
NIPS_TS_Water"""

ucr_index1=None
ucr_aug_index1=None

def get_bias(data,index,aff=0):
    others_bias_name1 = bias_name.split("\n")
    others_pre_bias1 = others_pre_bias.split("\n")
    others_aff_pre_bias1 = others_aff_pre_bias.split("\n")

    global ucr_index1,ucr_aug_index1
    if ucr_index1 is None:
        ucr_index1 = ucr_index.split("\n")
        ucr_index1 = {int(x):i for i,x in enumerate(ucr_index1)}
    if ucr_aug_index1 is None:
        ucr_aug_index1 = ucr_aug_index.split("\n")
        ucr_aug_index1 = {int(x):i for i,x in enumerate(ucr_aug_index1)}
   
    if data == "UCR":
        new_index = ucr_index1[index]
        if aff:
            return float(ucr_aff_pre_bias.split("\n")[new_index])
        else:
            return float(ucr_pre_bias.split("\n")[new_index])

    elif data == "UCR_AUG":
        new_index = ucr_aug_index1[index]
        if aff:
            return float(ucr_aug_aff_pre_bias.split("\n")[new_index])
        else:
            return float(ucr_aug_pre_bias.split("\n")[new_index])
        
    else:
        new_index = others_bias_name1.index(data)
        if aff:
            return float(others_aff_pre_bias1[new_index])
        else:
            return float(others_pre_bias1[new_index])



def combine_all_evaluation_scores_with_bias(pred_labels, y_test, anomaly_scores, full=True, data_name="UCR", index=1):
    events_pred = convert_vector_to_events(pred_labels) # [(4, 5), (8, 9)]
    events_gt = convert_vector_to_events(y_test)     # [(3, 4), (7, 10)]
    Trange = (0, len(y_test))
    
    true_events = get_events(y_test)
    accuracy, precision, recall, f1_score_ori, f05_score_ori = get_accuracy_precision_recall_fscore(y_test, pred_labels)
    print(f'f1_score_ori: {f1_score_ori}')
    print(f'f05_score_ori: {f05_score_ori}')
    bias_pre = get_bias(data_name, index, aff=0)
    bias_aff_pre = get_bias(data_name, index, aff=1)
    N_precision = (precision - bias_pre)/(1-bias_pre)
    N_F1 = 2*abs(N_precision)*recall/(abs(N_precision)+recall)
    if N_precision < 0:
        N_F1 *= -1
    f1_score_pa = get_point_adjust_scores(y_test, pred_labels, true_events)[5]
    print(f'f1_score_pa: {f1_score_pa}')
    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(pred_labels, y_test)
    print(f'pa_accuracy, pa_precision, pa_recall, pa_f_score:')
    print(pa_accuracy, pa_precision, pa_recall, pa_f_score)
    range_f_score = customizable_f1_score(y_test, pred_labels)
    _, _, f1_score_c = get_composite_fscore_raw(y_test, pred_labels,  true_events, return_prec_rec=True)
    precision_k = precision_at_k(y_test, anomaly_scores, pred_labels)
    point_auc = point_wise_AUC(anomaly_scores, y_test)
    range_auc = Range_AUC(anomaly_scores, y_test)
    MCC_score = MCC(y_test, pred_labels)

    if full:
        
        affiliation = pr_from_events(events_pred, events_gt, Trange)
        # results = get_range_vus_roc(y_test, pred_labels, 100) # slidingWindow = 100 default
        results = get_range_vus_roc(anomaly_scores, y_test, 100) # slidingWindow = 100 default

        UAff_Pre = (affiliation['precision'] - bias_aff_pre)/(1-bias_aff_pre)
        UAff_F1 = 2*abs(UAff_Pre)*affiliation['recall']/(abs(UAff_Pre)+affiliation['recall'])
        if UAff_Pre < 0:
            UAff_F1 *= -1

        NAff_Pre = (affiliation['precision'] - 0.5)/(1-0.5)
        NAff_F1 = 2*abs(NAff_Pre)*affiliation['recall']/(abs(NAff_Pre)+affiliation['recall'])
        if NAff_Pre < 0:
            NAff_F1 *= -1

        Aff_F1 = 2*affiliation['precision']*affiliation['recall']/(affiliation['precision']+affiliation['recall'])
    
        score_list = {"f1_score_ori": f1_score_ori, 
                      "accuracy":accuracy,
                      "precision":precision,
                      "recall":recall,
                    "f05_score_ori" : f05_score_ori, 
                    "f1_score_pa": f1_score_pa,
                    "pa_accuracy":pa_accuracy, 
                    "pa_precision":pa_precision, 
                    "pa_recall":pa_recall, 
                    "pa_f_score":pa_f_score,
                    "range_f_score": range_f_score,
                    "f1_score_c": f1_score_c, 
                    "precision_k": precision_k,
                    "point_auc": point_auc,
                    "range_auc": range_auc, 
                    "MCC_score":MCC_score, 
                    "Affiliation precision": affiliation['precision'], 
                    "Affiliation recall": affiliation['recall'],
                    "R_AUC_ROC": results["R_AUC_ROC"], 
                    "R_AUC_PR": results["R_AUC_PR"],
                    "VUS_ROC": results["VUS_ROC"], 
                    "VUS_PR": results["VUS_PR"],
                    "Aff_F1": Aff_F1,
                    "NAff_Pre": NAff_Pre,
                    "NAff_F1": NAff_F1,
                    "N_Pre": N_precision,
                    "N_F1": N_F1,
                    "UAff_Pre": UAff_Pre,
                    "UAff_F1": UAff_F1
        }

    else:
        score_list = {"f1_score_ori": f1_score_ori,
                      "accuracy":accuracy,
                      "precision":precision,
                      "recall":recall, 
                    "f05_score_ori" : f05_score_ori, 
                    "f1_score_pa": f1_score_pa,
                    "pa_accuracy":pa_accuracy, 
                    "pa_precision":pa_precision, 
                    "pa_recall":pa_recall, 
                    "pa_f_score":pa_f_score,
                    "range_f_score": range_f_score,
                    "f1_score_c": f1_score_c, 
                    "precision_k": precision_k,
                    "point_auc": point_auc,
                    "range_auc": range_auc, 
                    "MCC_score":MCC_score, 
                    # "Aff_F1": Aff_F1,
                    # "NAff_Pre": NAff_Pre,
                    # "NAff_F1": NAff_F1,
                    "N_Pre": N_precision,
                    "N_F1": N_F1
        }
    
    return score_list



def combine_all_evaluation_scores(pred_labels, y_test, anomaly_scores, full=True):
    events_pred = convert_vector_to_events(pred_labels) # [(4, 5), (8, 9)]
    events_gt = convert_vector_to_events(y_test)     # [(3, 4), (7, 10)]
    Trange = (0, len(y_test))
    
    true_events = get_events(y_test)
    accuracy, precision, recall, f1_score_ori, f05_score_ori = get_accuracy_precision_recall_fscore(y_test, pred_labels)
    print(f'f1_score_ori: {f1_score_ori}')
    print(f'f05_score_ori: {f05_score_ori}')
    f1_score_pa = get_point_adjust_scores(y_test, pred_labels, true_events)[5]
    print(f'f1_score_pa: {f1_score_pa}')
    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(pred_labels, y_test)
    print(f'pa_accuracy, pa_precision, pa_recall, pa_f_score:')
    print(pa_accuracy, pa_precision, pa_recall, pa_f_score)
    range_f_score = customizable_f1_score(y_test, pred_labels)
    _, _, f1_score_c = get_composite_fscore_raw(y_test, pred_labels,  true_events, return_prec_rec=True)
    precision_k = precision_at_k(y_test, anomaly_scores, pred_labels)
    point_auc = point_wise_AUC(anomaly_scores, y_test)
    range_auc = Range_AUC(anomaly_scores, y_test)
    MCC_score = MCC(y_test, pred_labels)

    if full:
        
        affiliation = pr_from_events(events_pred, events_gt, Trange)
        # results = get_range_vus_roc(y_test, pred_labels, 100) # slidingWindow = 100 default
        results = get_range_vus_roc(anomaly_scores, y_test, 100) # slidingWindow = 100 default

    
        score_list = {"f1_score_ori": f1_score_ori, 
                      "accuracy":accuracy,
                      "precision":precision,
                      "recall":recall,
                    "f05_score_ori" : f05_score_ori, 
                    "f1_score_pa": f1_score_pa,
                    "pa_accuracy":pa_accuracy, 
                    "pa_precision":pa_precision, 
                    "pa_recall":pa_recall, 
                    "pa_f_score":pa_f_score,
                    "range_f_score": range_f_score,
                    "f1_score_c": f1_score_c, 
                    "precision_k": precision_k,
                    "point_auc": point_auc,
                    "range_auc": range_auc, 
                    "MCC_score":MCC_score, 
                    "Affiliation precision": affiliation['precision'], 
                    "Affiliation recall": affiliation['recall'],
                    "R_AUC_ROC": results["R_AUC_ROC"], 
                    "R_AUC_PR": results["R_AUC_PR"],
                    "VUS_ROC": results["VUS_ROC"], 
                    "VUS_PR": results["VUS_PR"]
        }

    else:
        score_list = {"f1_score_ori": f1_score_ori,
                      "accuracy":accuracy,
                      "precision":precision,
                      "recall":recall, 
                    "f05_score_ori" : f05_score_ori, 
                    "f1_score_pa": f1_score_pa,
                    "pa_accuracy":pa_accuracy, 
                    "pa_precision":pa_precision, 
                    "pa_recall":pa_recall, 
                    "pa_f_score":pa_f_score,
                    "range_f_score": range_f_score,
                    "f1_score_c": f1_score_c, 
                    "precision_k": precision_k,
                    "point_auc": point_auc,
                    "range_auc": range_auc, 
                    "MCC_score":MCC_score, 
        }
    
    return score_list



def my_combine_all_evaluation_scores(anomaly_scores, y_test, ar=1.0):
    thrd = np.percentile(anomaly_scores,(100-ar))
    pred_labels = anomaly_scores > thrd
    events_pred = convert_vector_to_events(y_test) # [(4, 5), (8, 9)]
    events_gt = convert_vector_to_events(pred_labels)     # [(3, 4), (7, 10)]
    Trange = (0, len(y_test))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    true_events = get_events(y_test)
    _, _, _, f1_score_ori, f05_score_ori = get_accuracy_precision_recall_fscore(y_test, pred_labels)
    f1_score_pa = get_point_adjust_scores(y_test, pred_labels, true_events)[5]
    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(pred_labels,y_test)
    range_f_score = customizable_f1_score(y_test, pred_labels)
    _, _, f1_score_c = get_composite_fscore_raw(y_test, pred_labels,  true_events, return_prec_rec=True)
    precision_k = precision_at_k(y_test, anomaly_scores, pred_labels)
    point_auc = point_wise_AUC(anomaly_scores, y_test)
    # range_auc = Range_AUC(pred_labels, y_test)
    range_auc = 0#Range_AUC(pred_labels, y_test)
    MCC_score = MCC(y_test, pred_labels)
    results = get_range_vus_roc(y_test, pred_labels, 100) # slidingWindow = 100 default

    
    score_list = {"f1_score_ori": f1_score_ori, 
                  "f05_score_ori" : f05_score_ori, 
                  "f1_score_pa": f1_score_pa,
                  "pa_accuracy":pa_accuracy, 
                  "pa_precision":pa_precision, 
                  "pa_recall":pa_recall, 
                  "pa_f_score":pa_f_score,
                  "range_f_score": range_f_score,
                  "f1_score_c": f1_score_c, 
                  "precision_k": precision_k,
                  "point_auc": point_auc,
                  "range_auc": range_auc, 
                  "MCC_score":MCC_score, 
                  "Affiliation precision": affiliation['precision'], 
                  "Affiliation recall": affiliation['recall'],
                  "R_AUC_ROC": results["R_AUC_ROC"], 
                  "R_AUC_PR": results["R_AUC_PR"],
                  "VUS_ROC": results["VUS_ROC"], 
                  "VUS_PR": results["VUS_PR"]}
    
    return score_list


def main():
    y_test = np.zeros(100)
    y_test[10:20] = 1
    y_test[50:60] = 1
    pred_labels = np.zeros(100)
    pred_labels[15:17] = 1
    pred_labels[55:62] = 1
    anomaly_scores = np.zeros(100)
    anomaly_scores[15:17] = 0.7
    anomaly_scores[55:62] = 0.6
    pred_labels[51:55] = 1
    true_events = get_events(y_test)
    scores = combine_all_evaluation_scores(y_test, pred_labels, anomaly_scores)
    # scores = test(y_test, pred_labels)
    for key,value in scores.items():
        print(key,' : ',value)



ucr_index="""1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
250"""

ucr_pre_bias = """0.014623821
0.008137409
0.008268155
0.036598156
0.000570629
0.013081922
0.013066202
0.01572375
0.018900762
0.028455285
0.021972753
0.010975323
0.013434633
0.006972112
0.002794385
0.008828307
0.005022962
0.015641477
0.034850746
0.034770489
0.014970194
0.022228101
0.02209175
0.015494363
0.008169224
0.032724192
0.003632866
0.021786063
0.022490803
0.003238177
0.038885696
0.057946642
0.065973719
0.025082261
0.03052609
0.039499055
0.005187533
0.003618386
0.003133902
0.006161834
0.008208923
0.012501409
0.013013815
0.019681951
0.027790279
0.010293418
0.007241374
0.015458996
0.005411255
0.021121503
0.002041947
0.021429125
0.047332956
0.018252705
0.005936843
0.004546025
0.007739938
0.003157993
0.010151548
0.010294118
0.015361185
0.004805232
0.015495834
0.019922824
0.005801044
0.011354861
0.017005821
0.018292683
0.000478807
0.008463542
0.007034837
0.008251959
0.012565316
0.016258299
0.018257019
0.008582066
7.7211E-05
0.00325388
1.54933E-05
1.09506E-05
1.67194E-05
0.000745538
0.005349608
0.006734007
0.004429379
0.008631024
0.013160568
0.002296915
0.011377197
0.006734007
3.60038E-05
0.013893288
0.026587845
0.022626072
0.010632273
0.011126943
0.002868072
0.031799332
0.003194635
0.008310249
0.013402042
0.012855415
0.003332757
0.015164973
0.018369318
0.032067963
0.014461964
0.008499887
0.00730912
0.029129043
0.000428661
0.018292683
0.030487805
0.014089207
0.01893886
0.040650407
0.014557222
0.00754702
0.010379915
0.005940878
0.00218414
0.01369126
0.00421105
0.005434923
0.010749072
0.038469573
0.013757508
0.019492833
0.021076504
0.017738359
0.017985612
0.030908703
0.004062346
0.021319466
0.022860514
0.004897837
0.053670397
0.058206822
0.064558366
0.038943496
0.023931268
0.035140785
0.003413134
0.004414444
0.00179447
0.007098338
0.006215734
0.011127719
0.017752578
0.019452307
0.023875502
0.008265113
0.00688429
0.021119734
0.003533967
0.014472242
0.00487277
0.013661112
0.048922182
0.041722001
0.007012654
0.006055343
0.004559978
0.003991132
0.00816546
0.010089676
0.015108532
0.003015161
0.016964203
0.019910719
0.00565999
0.010767219
0.012787992
0.017381211
0.000209326
0.008163061
0.009351227
0.009786723
0.015361953
0.015339812
0.01855224
0.008445655
3.32491E-05
0.001749481
1.40965E-05
2.6889E-05
5.80046E-06
0.000719328
0.005490895
0.006012506
0.004867872
0.009118827
0.012074873
0.004684719
0.0107047
0.004543047
1.46548E-05
0.001895919
0.003194888
0.002883441
0.007575758
0.003601569
0.005494505
0.003644256
0.009615385
0.00239521
0.001944352
0.001753277
0.00224918
0.003116359
0.004740235
0.00275381
0.002578243
0.002994049
0.001285097
0.003964519
0.005708505
0.003862769
0.006329092
0.005175414
0.003749743
0.004491338
0.002563473
0.001572877
0.000429519
0.002838504
0.00659861
0.007050006
1.35634E-05
3.0154E-05
3.33187E-05
0.001508621
0.010674331
0.003204489
0.00018662
0.00023427
0.002723917
0.002146041
0.000588398
0.001351163
0.001698807
0.002635542
0.001367504
0.001780676
0.01170184
0.009259259"""

ucr_aff_pre_bias = """0.498802088
0.502617427
0.499603769
0.49722261
0.501915291
0.488328374
0.517923548
0.500966798
0.494536926
0.525193873
0.502601706
0.474331401
0.497118447
0.502800021
0.502571441
0.500432415
0.4998778
0.485135423
0.517077819
0.501955522
0.507619415
0.516649228
0.503658616
0.496727387
0.50212866
0.500333156
0.499548126
0.501018764
0.504423802
0.493675512
0.50294734
0.500468569
0.510017667
0.499974193
0.503765487
0.516306623
0.502936036
0.500570537
0.499507236
0.501526389
0.50061758
0.498331265
0.499573992
0.503261906
0.498837403
0.522198167
0.499418213
0.505542062
0.488784645
0.517281642
0.494084158
0.500040863
0.50031116
0.511048481
0.503490819
0.496355086
0.509550147
0.508916216
0.501911702
0.49957333
0.499501322
0.504993878
0.499867774
0.500256058
0.500723891
0.519989554
0.490822055
0.51692944
0.508970448
0.491867291
0.502607955
0.497331053
0.503202288
0.49909023
0.499095168
0.500090647
0.500621502
0.500446554
0.499825962
0.49970713
0.500256947
0.498564592
0.498565428
0.491917366
0.500221043
0.499690257
0.498668406
0.497906658
0.500473784
0.492734211
0.501279475
0.500405915
0.496587002
0.50360828
0.498843159
0.500016951
0.495109481
0.500890742
0.499853969
0.495425179
0.492891599
0.502938136
0.499870752
0.501237917
0.493933756
0.478940958
0.500199205
0.499019042
0.499209106
0.50049306
0.501260228
0.521089135
0.492966815
0.503143876
0.500421287
0.544387941
0.498843419
0.499946113
0.499717576
0.50456792
0.503512087
0.510488429
0.500633578
0.498405208
0.506596522
0.500635194
0.508174465
0.509863172
0.502593703
0.499570995
0.485202853
0.501891794
0.499495729
0.50497915
0.499634298
0.496516536
0.50410988
0.503272097
0.504364178
0.517521322
0.502409073
0.512785042
0.49618917
0.495011324
0.499860858
0.499746886
0.501266197
0.499530218
0.519744186
0.495716973
0.501452724
0.495667063
0.500590691
0.505818716
0.493734098
0.503491682
0.485108999
0.499931067
0.498315028
0.534768042
0.506921331
0.50509864
0.502863953
0.495411612
0.505581243
0.500673158
0.500066397
0.501016112
0.500039005
0.500957354
0.491441864
0.504204693
0.499925617
0.477622496
0.497832427
0.508660475
0.499880396
0.502520728
0.505621249
0.500608375
0.501603412
0.500777525
0.500081756
0.503045833
0.500316968
0.499231218
0.499290999
0.501044066
0.518082855
0.48124695
0.504109889
0.497847851
0.502711221
0.500533601
0.500181899
0.496922045
0.500392035
0.509144298
0.508269274
0.507017271
0.497511937
0.497827331
0.510876879
0.502301449
0.517969639
0.499279024
0.498676479
0.497824648
0.497909337
0.498522371
0.509037355
0.498126083
0.500268669
0.500219408
0.501482211
0.504161578
0.5005033
0.502474271
0.494173325
0.497321708
0.501626572
0.501967469
0.500576046
0.499200315
0.499340394
0.50156489
0.497666163
0.505370132
0.249425269
0.49967206
0.499512065
0.499825436
0.499003833
0.495092163
0.501089185
0.499584817
0.500018647
0.500431562
0.502395071
0.501938463
0.496909662
0.501535535
0.488629852
0.501691501
0.485096222
0.548010609"""

ucr_aug_pre_bias="""0.094457696
0.067587098
0.071709702
0.099431582
0.105092726
0.107924451
0.103083427
0.102212341
0.112335288
0.098505014
0.207750094
0.251704832
0.257804344
0.187153403
0.022669646
0.169445492
0.166341163
0.180305761
0.144242587
0.19128002
0.136526285
0.153548177
0.144977689
0.598901434
0.424511666
0.379281241
0.40439471
0.48199778
0.470117188
0.603210261
0.558600109
0.440040692
0.681155881
0.435320448
0.486163863
0.624745625
0.063210681
0.063156604
0.061598426
0.056957344
0.070919526
0.068422677
0.079515894
0.127406613
0.122042388
0.148540315
0.109577566
0.100373561
0.108150692
0.094705583
0.10795578
0.15215537
0.163415328
0.039383234
0.029291707
0.033281814
0.034159813
0.150329732
0.148103053
0.168446752
0.138025839
0.156865102
0.185932381
0.054181582
0.061885779
0.084863455
0.056294396
0.058703256
0.036828967
0.04622768
0.040613073
0.063386431
0.079214233
0.028065871
0.021625538
0.030654953
0.017287372
0.018405353
0.020310029
0.023506678
0.019473502
0.020684095
0.018555929
0.064128814
0.094190664
0.117341822
0.085611924
0.03858426
0.032918718
0.023918282
0.093019503
0.081721276
0.115323978
0.133121541
0.166053121
0.35190234
0.389175992
0.060172221
0.06784114
0.108014038
0.033715296
0.155423697
0.158281616
0.077944658
0.01716608
0.094093916
0.079959537
0.078647696
0.085432255
0.11020459
0.116732625
0.115981423
0.107628539
0.107214628
0.122719564
0.195660214
0.23401246
0.260060616
0.180775277
0.023958087
0.174409728
0.150842183
0.182689066
0.159363653
0.181122689
0.167782424
0.144178666
0.159517656
0.601257274
0.476397179
0.367203108
0.373591048
0.47254162
0.478076172
0.477050617
0.451220836
0.400976602
0.606186406
0.434628688
0.390830168
0.674399674
0.057796091
0.052709952
0.056818778
0.066585315
0.069007065
0.073741246
0.099709991
0.115119683
0.117623655
0.145536368
0.105165375
0.089257355
0.106331293
0.082896301
0.101516238
0.123212522
0.15424503
0.038991406
0.031485946
0.032859496
0.033185337
0.159045882
0.141524822
0.169781816
0.145825118
0.153575551
0.178991285
0.054764287
0.068594198
0.093718376
0.046904789
0.059279843
0.037268717
0.045184007
0.04106946
0.064026128
0.071872099
0.028034643
0.021477399
0.030908079
0.017110553
0.018287999
0.020532094
0.023679743
0.0191478
0.020249677
0.018981367
0.064078844
0.101120687
0.118368264
0.087865609
0.043130688
0.033011523
0.024008399
0.028238364
0.041889529
0.036165749
0.034429289
0.042122711
0.081356929
0.081963427
0.091427864
0.094618106
0.082081744
0.083010134
0.0501051
0.049210593
0.050034812
0.051449189
0.060261012
0.052675642
0.008666866
0.008770529
0.007742263
0.007638654
0.008094472
0.009695483"""

ucr_aug_aff_pre_bias = """0.509504932
0.506962749
0.507509748
0.525424982
0.509223248
0.5108012
0.520062741
0.536637689
0.547955767
0.515401912
0.552497162
0.562524124
0.558825649
0.544801704
0.501472835
0.534742772
0.53863599
0.533134914
0.532895361
0.533992442
0.517887545
0.548836576
0.530424353
0.664309184
0.605875131
0.577360482
0.606211368
0.656151324
0.626236722
0.687618054
0.683148446
0.60588667
0.663379953
0.60786314
0.624835395
0.718597023
0.50355575
0.507437128
0.503716486
0.505569866
0.505914053
0.518108101
0.506784397
0.537329224
0.531345552
0.525250919
0.525643901
0.51233622
0.515995696
0.51041399
0.517551817
0.512807503
0.527218525
0.505089145
0.501105102
0.503309338
0.501798165
0.552157367
0.521400798
0.534555191
0.513139253
0.540807343
0.538293875
0.503000844
0.506664443
0.532846995
0.507447684
0.505372787
0.501494165
0.503250398
0.501085056
0.503160927
0.507458733
0.521377836
0.511209502
0.503006933
0.5053418
0.50196601
0.507358414
0.501534234
0.502683453
0.502435595
0.502392062
0.502716408
0.526961812
0.514678945
0.519603731
0.501673374
0.507668509
0.50238258
0.506030188
0.507197286
0.512626899
0.536110112
0.525199446
0.570874408
0.599229788
0.508858308
0.513832615
0.522826392
0.508003841
0.524295887
0.552836046
0.516416206
0.501608646
0.506710042
0.512596265
0.509068047
0.504391828
0.506795374
0.531374049
0.538594405
0.510780049
0.510389503
0.52236483
0.556001037
0.549780402
0.574621371
0.552710724
0.501626164
0.529430907
0.519067584
0.52648409
0.53697416
0.526462813
0.527611654
0.550594315
0.530182558
0.669666322
0.61516342
0.601550286
0.632573594
0.641648616
0.622025036
0.605571122
0.625925545
0.598965089
0.63932376
0.619195674
0.60635591
0.740194376
0.510101417
0.503622294
0.503614489
0.512543379
0.504330044
0.506628432
0.510052656
0.511749749
0.513055589
0.562988042
0.525734094
0.507301575
0.514702775
0.522294207
0.526387374
0.510354486
0.551007454
0.503231611
0.504080762
0.500215003
0.504282213
0.520695135
0.518066434
0.522352022
0.530939359
0.528328854
0.536082635
0.519217285
0.527022616
0.51097177
0.510481219
0.503517511
0.505270162
0.53694062
0.502636409
0.502789642
0.506065797
0.505477955
0.501000959
0.506543863
0.501027692
0.502885592
0.502248315
0.501282674
0.505352607
0.503045716
0.501466614
0.50565587
0.512532831
0.521358862
0.529966976
0.514146493
0.503859317
0.502765054
0.502530907
0.522742503
0.502992627
0.503581969
0.507500925
0.526071324
0.511497957
0.529439794
0.530988198
0.515028807
0.515255764
0.510032652
0.514532909
0.503810849
0.503103389
0.50666335
0.504870006
0.502801064
0.500251656
0.500831949
0.500644486
0.499077675
0.500297373"""

ucr_aug_index="""1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
242
243
244
245
246
247"""
    
if __name__ == "__main__":
    # main()
    test_bias()
