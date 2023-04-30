# Import Packages
import numpy as np
import pandas as pd
import random 
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys

# Parameters
alpha = 0.01
s = 200000
freq_min = 0.2
freq_max = 0.7
qtn = 1
b = 0.5 
h = 0.25
d= 20

# Create temporary dataframe to store results
n_simulations = 1
n_sig_snps = np.zeros(n_simulations)
n_sig_snps_fdr = np.zeros(n_simulations)
n_qtn_found = np.zeros(n_simulations)
n_qtn_found_fdr = np.zeros(n_simulations)
results_csv_filename = "data-nsnps={}-q={}-h={}.csv".format(s, qtn, h)

# freq_array creates the allele frequency for each snp based on freq_min and frq_max
freq_array = [random.uniform(freq_min, freq_max) for _ in range(s)]

# initialize drone matrix
drone_geno = np.zeros((d, s))

# loop over haplotype of each drone and select allele based on freq_array
for j in range(d):
    haplo = np.zeros(s, dtype=int)

    # loop over SNPS
    for i in range(s):
        haplo[i] = np.random.binomial(1, freq_array[i]) # randomly pick 1 allele using binomal distribution given allele frequency for specific SNP

    drone_geno[j, :] = haplo


#causal snp module
qtn_pos = random.sample(range(s), qtn)

#create environmental variation 
g = qtn*b/2
e=(g-h*g)/h

# initilize pheno as NumPy array of zeros
pheno = np.zeros(d)
# create an array of QTN positions
qtn_array = np.array(drone_geno[:, qtn_pos])
# boolean indexing to update pheno for each drone
pheno += np.sum(qtn_array ==1, axis=1)*b
pheno -= np.sum(qtn_array ==0, axis=1)*b

# create mothers contribution
# grab the qtn position and make moms qtn array
qtn_pos = list(qtn_pos)
qtn_array_mom = []
for j in range(len(qtn_pos)):
    qtn_array_mom.append(freq_array[qtn_pos[j]])

# create moms genotype from the mom qtn array    
mom_geno_alleles = []
mom_geno = []
for qtn in qtn_array_mom:
    allele1 = np.random.binomial(n=1, p=qtn, size=1)[0]
    allele2 = np.random.binomial(n=1, p=qtn, size=1)[0]
    mom_geno_alleles.append([allele1, allele2])
mom_geno = np.array(mom_geno_alleles)

# calculate the phenotype for each row
mom_pheno = np.where(np.all(mom_geno == [0, 0], axis=1), -1, 
                     np.where(np.all(mom_geno == [0, 1], axis=1) | np.all(mom_geno == [1, 0], axis=1), 0, 
                              np.where(np.all(mom_geno == [1, 1], axis=1), 1, None)))
# number of workers per father                              
w = 10  

# Attach the phenotype as a new column to the genotype matrix
drone_geno_pheno = np.concatenate((drone_geno, pheno.reshape(-1, 1)), axis=1)
ids = np.arange(1, d+1).reshape(-1, 1)
drone_geno_pheno_ids = np.concatenate((drone_geno_pheno, ids), axis=1)

workers = np.array([drone_geno_pheno_ids[0,:]])

for j in range(0, d):
    for i in range(0, w):
        workers = np.vstack((drone_geno_pheno_ids[j,:], workers))

# Remove duplicated first row
workers = workers[:-1, :]


# In[8]:


# enviormental variation
worker_env = np.random.normal(loc=e, scale=1, size=(w * d))

# add moms phenotype to workers array
mom_pheno = np.asarray(mom_pheno)
workers_per_drone = w * d
mom_sum = np.zeros((1, workers_per_drone))
# not sure why i made this so wack - moms phenotype function
def compute_sums(mom_pheno, mom_sum):
    # Compute the sum for each column of mom_sum and add it to the corresponding column of mom_sum
    for i in range(mom_sum.shape[1]):
        mom_sum[0, i] += np.sum(np.where(mom_pheno == 1, b, np.where(mom_pheno == -1, -b, np.random.choice([-b, b]))))
    return mom_sum

# Call the function to update mom_sum
mom_sum = compute_sums(mom_pheno, mom_sum)

# finalize worker matrix 
workers[:,-2] = workers[:,-2] + worker_env + mom_sum

# add columns to df_workers
df_workers = pd.DataFrame(workers)
new_cols = df_workers.columns.to_list()
new_cols = [f'snp_{j}' for j in new_cols]
new_cols[-2] = "phenotype"
new_cols[-1] = "ID"
df_workers.columns = new_cols
# display(df_workers)

# Initialize the variables
n_sig_snps = 0
n_sig_snps_fdr = 0
n_qtn_found = 0
n_qtn_found_fdr = 0

# Compute a mixed effects linear regression for each SNP
# Added try/except loop to remove matrix error, and prevent code from haulting
results_df = []
for snp_pos_j in range(s):
    # Make a copy of the DataFrame to use in the model fitting
    df_copy = df_workers.copy()
    try:
        # Create the formula dynamically based on the remaining columns (Why??)
        formula = f"phenotype ~ snp_{snp_pos_j} + 1"
        corlmer = smf.mixedlm(formula, df_copy, groups=df_copy["ID"]).fit()
    except np.linalg.LinAlgError:
        print(f"Singular matrix encountered for SNP {snp_pos_j}")
        df_copy = df_copy.drop(f'snp_{snp_pos_j}', axis=1)
        # Skip this SNP and process the next one.
        continue

    pvalues = corlmer.pvalues
    corrected_pvalues = multipletests(pvalues, method='Bonferroni')[1]  # method set to 'Bonferroni' or 'fdr_bh'
    
    # Number of SNPs detected
    if pvalues[1] < alpha:
        n_sig_snps += 1
    if corrected_pvalues[1] < alpha:
        n_sig_snps_fdr += 1
    if snp_pos_j in qtn_pos and pvalues[1] < alpha:
        n_qtn_found += 1
    if snp_pos_j in qtn_pos and corrected_pvalues[1] < alpha:
        n_qtn_found_fdr += 1
    
    # Append results list to DataFrame
    results_df.append([f'snp_{snp_pos_j}', pvalues[1], corrected_pvalues[1]])
# Print the final results table
results_df = pd.DataFrame(results_df, columns=['snp','pvalue','corrected_pvalue'])
# print(results_df)
# Write DataFrame to CSV file
results_df.to_csv("{}_{}".format(sim_number, results_csv_filename), index=False)
# Print counts
print(f"Number of significant SNPs: {n_sig_snps:,}")
print(f"Number of significant SNPs (FDR-corrected): {n_sig_snps_fdr:,}")
print(f"Number of QTNs found: {n_qtn_found:,}")
print(f"Number of QTNs found (FDR-corrected): {n_qtn_found_fdr:,}")
print(f"QTN position: {qtn_pos}")

