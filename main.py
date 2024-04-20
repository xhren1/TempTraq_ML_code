from preprocess import *
from plot_cluster import *
from sklearn.metrics import roc_curve, auc

# Parameters   
temp_data_path = "[Please input your path to the temperature data file here]"
fever_start_data_path = "[Please input your path to the fever start data file here]"
fever_cause_data_path = "[Please input your path to the fever cause data file here]"
cohort = "HCT"
before = 4
after = 4
original_percent_threshold = 0.7
cluster_number = 3
missing_points = True # True: deal with missing points(for the first time run), False: not deal with missing points(if you have a full_ttemp.csv file in the current directory)

# Load the data
if missing_points:
    temp_df = deal_with_missing_points(temp_data_path,cohort)    
else:
    temp_df = pd.read_csv('./full_ttemp.csv')
    
temp_df = interpolation_smooth(temp_df,fever_start_data_path,cohort)
temp_array, qualified_temp_label, qualified_temp_percent, true_label, qualified_fever_causes = filter_data(temp_df,before,after,original_percent_threshold,fever_cause_data_path)

# Plot all the data before clustering
print("Plot the data before clustering...")
plt.rcParams['figure.figsize'] = [20, 6.4]
plt.rcParams['figure.figsize'] = [40, (int(len(temp_array)/5)+1)*4]
data_plot(temp_array,qualified_temp_label)
plt.savefig('./30_total_fevers.svg', format='svg')
plt.close()

# cluster number decision with elbow method
print("Cluster number decision with elbow method...")
plt.rcParams['figure.figsize'] = [5, 2]
cluster_number_decision(temp_array,2,9,10)
plt.savefig('./with_unclear_elbow_method.svg', format='svg')
plt.close()

# clustering
print("Clustering...")
print(f"Cluster number: {cluster_number}")
plt.rcParams['figure.figsize'] = [20, 6.4]
predict_label,cluster_centers,model,silhouette = DTW_KMeans_clustering(train_data= temp_array, cluster_number=cluster_number,seed=10)
print("silhouette score: {:.2f}".format(silhouette))
plt.savefig('./with_unclear_fevers_cluster_result.svg', format='svg')
plt.close()

# ROC curve is not designed for clustering, but we can use it to evaluate the clustering result
# because 0 is non-infection and 1 is infection, we need to make cluster with the most infection to the largest number to draw the ROC curve
# make all 0 to 3
predict_label[predict_label == 0] = 3
predict_label

print("Plot the results...")
fpr, tpr, _ = roc_curve(true_label, predict_label)
roc_auc = auc(fpr, tpr)
plt.rcParams['figure.figsize'] = [10, 10]
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=4, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axvline(x=0.6, ymin=1, color='red', linestyle='--',lw=0.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity')
plt.ylabel('Senstivity')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('./with_unclear_ROC.svg', format='svg')
plt.close()

# combine the cluster result with the fever causes
cluster_result ={}
for i,label in enumerate(predict_label):
    if label not in cluster_result:
        cluster_result[label] = [qualified_temp_label[i]]
    else:
        cluster_result[label].append(qualified_temp_label[i])

#write the cluster result to csv file
result = pd.DataFrame({'MaskID':[i for (i,j) in qualified_temp_label], 'Time_DPI':[j for (i,j) in qualified_temp_label], 'Category': qualified_fever_causes, 'cluster result':predict_label+1})
result.to_csv("./result.csv", index=False)

# plot the number of fever events in each cluster(non-infection adn unclear seperated)(change with the result of your clustering)
plt.rcParams['figure.figsize'] = [20, 4]
fig, axs = plt.subplots(1,len(cluster_result.keys()))
#display the detail of each cluster
for key in range(len(cluster_result.keys())):
    
    # plot the number of fever events in each cluster with bar chart
    index_all = ['non-infection','infection', 'unclear']
    index = result[result['cluster result']==key+1]['Category'].value_counts().index
    value = result[result['cluster result']==key+1]['Category'].value_counts().values
    

    plt.subplot(1,len(cluster_result.keys()),key+1)
    plt.bar(index,value)
    plt.xlabel('Fever causes')
    plt.ylabel('Number of fever events')
    plt.ylim(0, 7.5)
    plt.title('Number of fever events in cluster {}'.format(key+1))

plt.savefig('./with_unclear_fever_causes.svg', format='svg')    
plt.close()

# plot the number of fever events in each cluster(non-infection adn unclear seperated)(to make the plot in the same sequence, I count the fever events in each cluster)
# please change the value if you are doing a different clustering
plt.rcParams['figure.figsize'] = [20, 4]
fig, axs = plt.subplots(1,len(cluster_result.keys()))
#display the detail of each cluster
color = ['cornflowerblue','sandybrown','lightcoral']
index_all = ['non-infection','infection', 'unclear']

###********** change it if you are doing a different clustering *********
value = [[0,7,0],[4,3,1],[4,6,5]]

for key in range(len(value)):
    
    # plot the number of fever events in each cluster with bar char

    plt.subplot(1,len(cluster_result.keys()),key+1)
    plt.bar(index_all,value[key],color=color)
    plt.xlabel('Fever causes')
    plt.ylabel('Number of fever events')
    plt.ylim(0, 9.5)
    plt.title('Number of fever events in cluster {}'.format(key+1))
    
plt.savefig('./with_unclear_fever_causes_colored_version.svg', format='svg')
plt.close()




# t-test for the number of fever events in each cluster with random clustering
# print("T-test for the number of fever events in each cluster with random clustering...")


print("Done!")