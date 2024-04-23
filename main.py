from preprocess import *
from plot_cluster import *
from stat_test import *
from scipy.stats import chi2_contingency

# Parameters   
temp_data_path = "[Please input your path to the temperature data file here]"
fever_start_data_path = "[Please input your path to the fever start data file here]"
fever_cause_data_path = "[Please input your path to the fever cause data file here]"
temp_data_path = "../input/TempTraq_Dataset.csv"
fever_start_data_path = "../input/TFeverStarts.csv"
fever_cause_data_path = "../input/4-17-19_With_PHI_HCT_result_with_exact_time_clinical_categories.csv"
cohort = "HCT"
before = 4
after = 4
original_percent_threshold = 0.7
cluster_number = 3
missing_points = False # True: deal with missing points(for the first time run), False: not deal with missing points(if you have a full_ttemp.csv file in the current directory)
num_iterations = 200

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
plt.rcParams['figure.figsize'] = [14, 6]
predict_label,cluster_centers,model,silhouette = DTW_KMeans_clustering(train_data= temp_array, cluster_number=cluster_number,seed=10)
print("silhouette score: {:.2f}".format(silhouette))
plt.savefig('./with_unclear_fevers_cluster_result.svg', format='svg')
plt.close()

print("Plot the results...")

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

# plot the number of fever events in each cluster
plt.rcParams['figure.figsize'] = [14, 6]
# Data
infection_cases = []
non_infection_cases = []
unclear_cases = []

for i in range(cluster_number):
    infection_cases.append(list(result[result['cluster result']==i+1]['Category']).count("infection"))
    non_infection_cases.append(list(result[result['cluster result']==i+1]['Category']).count("Other Adverse Event"))
    unclear_cases.append(list(result[result['cluster result']==i+1]['Category']).count("unclear"))
    
# Creating subplots with stacked bars for non-infection cases
fig, axs = plt.subplots(1, cluster_number, figsize=(12, 4), sharey=True)

# Plotting data with new stacked bar for unclear cases
for i, ax in enumerate(axs):
    ax.bar('Infection', infection_cases[i], color='blue', label='Infection' if i == 0 else "")
    ax.bar('Non-infection', non_infection_cases[i], color='orange', label='Non-infection' if i == 0 else "")
    ax.bar('Non-infection', unclear_cases[i], color='green', bottom=non_infection_cases[i], label='Unclear' if i == 0 else "")
    if i == 0:
        ax.set_ylabel('Number of Cases')
    if i == int((cluster_number)/2):
        ax.set_xlabel('Fever Causes')
    ax.set_title(f'cluster {i+1}')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')

# Adding a legend to explain the colors
fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
#save the figure with legend inside
plt.savefig('./with_unclear_fever_causes.svg', format='svg',bbox_inches='tight')

plt.close()

# t-test for the number of fever events in each cluster with random clustering
print("*"*80)
print("T-test for silhouette scores and accuracy between random clustering and DTW-KMeans clustering...")


stat_results = Parallel(n_jobs=-1)(delayed(compute_silhouette_and_accuracy)(i,temp_array,true_label,cluster_number) for i in range(num_iterations))

model_silhouette = []
random_silhouette = []
model_acc = []
random_acc = []
for model_score, random_score,model_accuracy,random_accuracy in stat_results:
    model_silhouette.append(model_score)
    random_silhouette.append(random_score)
    model_acc.append(model_accuracy)
    random_acc.append(random_accuracy)

# Plot the silhouette score distribution
plt.rcParams['figure.figsize'] = [8, 4.5]
plt.hist(model_silhouette, bins=10, alpha=0.5, label='Model')
plt.ylabel('Frequency')
plt.xlabel('Silhouette Score')
plt.title('Model Silhouette Score Distribution')
plt.savefig('./model_silhouette_score_histgram.svg', format='svg')
plt.close()

#qq plot
stats.probplot(model_silhouette, dist="norm", plot=plt)
plt.title('Model Silhouette Score Q-Q Plot')
plt.savefig('./model_silhouette_score_qq_plot.svg', format='svg')
plt.close()

plt.hist(random_silhouette, bins=10, alpha=0.5, label='Model')
plt.ylabel('Frequency')
plt.xlabel('Silhouette Score')
plt.title('Random Silhouette Score Distribution')
plt.savefig('./random_silhouette_score_histgram.svg', format='svg')
plt.close()

#qq plot
stats.probplot(random_silhouette, dist="norm", plot=plt)
plt.title('Random Silhouette Score Q-Q Plot')
plt.savefig('./random_silhouette_score_qq_plot.svg', format='svg')
plt.close()

t_statistic, p_value = stats.ttest_ind(model_silhouette, random_silhouette)
print("Silhouette score t-test:")
print(f"t-statistic: {t_statistic}, p-value: {p_value}")
print("-"*80)

# Plot the accuracy distribution
plt.rcParams['figure.figsize'] = [8, 4.5]
plt.hist(model_acc, bins=7, alpha=0.5, label='Model')
plt.ylabel('Frequency')
plt.xlabel('Accuracy Score')
plt.title('Model Accuracy Score Distribution')
plt.savefig('./model_accuracy_score_histgram.svg', format='svg')
plt.close()

# qq plot
stats.probplot(model_acc, dist="norm", plot=plt)
plt.title('Model Accuracy Score Q-Q Plot')
plt.savefig('./model_accuracy_score_qq_plot.svg', format='svg')
plt.close()

plt.hist(random_acc, bins=7, alpha=0.5, label='Model')
plt.ylabel('Frequency')
plt.xlabel('Accuracy Score')
plt.title('Random Accuracy Score Distribution')
plt.savefig('./random_accuracy_score_histgram.svg', format='svg')
plt.close()

# qq plot
stats.probplot(random_acc, dist="norm", plot=plt)
plt.title('Random Accuracy Score Q-Q Plot')
plt.savefig('./random_accuracy_score_qq_plot.svg', format='svg')
plt.close()

t_statistic, p_value = stats.ttest_ind(model_acc, random_acc)
print("Accuracy t-test:")
print(f"t-statistic: {t_statistic}, p-value: {p_value}")
print("-"*80)


observed = []
for i in range(cluster_number):
    observed.append([sum(true_label[predict_label == i] == 1),sum(true_label[predict_label == i] == 0)])
observed = np.array(observed).transpose()

chi2_stat, p_value, dof, expected = chi2_contingency(observed)

print("Chi-square Test for the number of fever causes in each cluster:")
print("Chi-square Statistic:", chi2_stat)
print("P-value:", p_value)
print("Degrees of Freedom:", dof)


print("Done!")