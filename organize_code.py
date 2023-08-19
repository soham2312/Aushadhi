# Import necessary libraries
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess the data
def preprocess_data(data):
    main_1=data.drop_duplicates(subset='patient_nbr',keep='first')
    med=['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    for i in med:

        main_1[i] = main_1[i].replace('No', pd.NA)
    data_med = main_1.dropna(subset=med, how='all')

    for i in med:
        data_med[i] = data_med[i].replace(pd.NA,'No')
    
    categorical_columns = ['gender', 'max_glu_serum', 'A1Cresult']

# Mapping for binary conversion
    binary_mapping = {
        'Male': 1, 
        'Female': 0,
        'Unknown/Invalid':0,  
        '>300': 1, 
        '>200': 1,  
        'Norm': 1,  
        '>8': 1,  
        'Norm': 1,  
        '>7': 1,  
        'None': 0,  
        np.nan:0.
    }

# Convert categorical variables to binary values within the same columns
    for col in categorical_columns:
        data_med[col] = data_med[col].map(binary_mapping)
    
    from itertools import chain
    new5=data_med

    category_mapping = {
        'circulatory': range(390, 460),
        'diabetes': [250],
        'digestive': range(520, 580),
        'genitourinary': range(580, 630),
        'injuries': range(800, 1000),
        'musculoskeletal': range(710, 740),
        'neoplasms': range(140, 240),
        'respiratory': range(460, 520),
        'other': chain(range(1,139),range(240,389),range(580,709),range(740,799))  # Add relevant ICD-9 codes for the 'other' category
    }
    # Create a dictionary to store disease categories
    disease_columns = {}

    # Create new columns for the specified disease categories
    disease_categories = ['circulatory', 'diabetes', 'digestive', 'genitourinary', 'injuries', 'musculoskeletal', 'neoplasms', 'respiratory', 'other']

    for col in ['diag_1', 'diag_2', 'diag_3']:
        for category in disease_categories:
            new_col_name = f'{category}'
            if new_col_name not in disease_columns:
                disease_columns[new_col_name] = new5[col].apply(lambda code: 1 if code in category_mapping[category] else 0)

    # Combine disease columns to create a single column for each category
    for category, column in disease_columns.items():
        new5[category] = column

    # Drop the original diagnosis columns
    new5.drop(['diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)

    race_column = 'race'
    race_categories = ['Caucasian', 'AfricanAmerican', 'Other', 'Asian', 'Hispanic']

    # Create new variables from the 'race' categories
    for category in race_categories:
        new_column_name = f'race_{category}'
        new5[new_column_name] = (new5[race_column] == category).astype(int)

    new5= new5.drop(columns=[race_column])
    age_mapping = {
        '[0-10)': 5,
        '[10-20)': 15,
        '[20-30)': 25,
        '[30-40)': 35,
        '[40-50)': 45,
        '[50-60)': 55,
        '[60-70)': 65,
        '[70-80)': 75,
        '[80-90)': 85,
        '[90-100)': 95
    }

    age_column = 'age'
    new5[age_column] = new5[age_column].map(age_mapping)

    # Convert '?' to pd.NA (null) values
    new5['weight'] = new5['weight'].replace('?', pd.NA)

    # Set up mean and standard deviation for generating random weights
    mean_diabetes_weight = 86
    std_diabetes_weight = 10

    # Generate random weight data for missing values
    np.random.seed(42)
    final_data = new5.copy()

    # Find indices of missing weights
    missing_weight_indices = final_data[final_data['weight'].isna()].index

    # Generate random weights
    num_missing_weights = len(missing_weight_indices)
    random_weights = np.random.randint(mean_diabetes_weight - std_diabetes_weight, mean_diabetes_weight + std_diabetes_weight + 1, num_missing_weights)

    # Update the DataFrame with the random weights
    final_data.loc[missing_weight_indices, 'weight'] = random_weights

    # Convert the 'weight' column to string type and then replace non-integer values
    final_data['weight'] = final_data['weight'].astype(str).str.extract('(\d+)').astype(float)

    columns_re=['encounter_id','patient_nbr','payer_code','medical_specialty','chlorpropamide','acarbose','miglitol','troglitazone','examide','citoglipton','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone','acetohexamide','tolbutamide','tolazamide']
    final_data=final_data.drop(columns=columns_re)

    final1=final_data.copy()
    drug_columns = ['metformin',
        'repaglinide', 'nateglinide', 'glimepiride', 'glipizide', 'glyburide',
        'pioglitazone', 'rosiglitazone', 'insulin', 'glyburide-metformin',]
    admin_threshold = 0.005  # 0.50%
    drug_admin_freq = (final1[drug_columns] != 'no').sum() / len(final1)

    # Filter out drugs based on the administration frequency threshold
    drugs_to_keep = drug_admin_freq[drug_admin_freq >= admin_threshold].index
    # Select only the columns of drugs to keep
    final1_filtered = final1[['gender', 'age', 'weight', 'admission_type_id',
        'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
        'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses', 'max_glu_serum', 'A1Cresult','change', 'diabetesMed', 'readmitted', 'circulatory', 'diabetes',
        'digestive', 'genitourinary', 'injuries', 'musculoskeletal',
        'neoplasms', 'respiratory', 'other', 'race_Caucasian',
        'race_AfricanAmerican', 'race_Other', 'race_Asian', 'race_Hispanic'] + list(drugs_to_keep)]
    
    final2=final1_filtered.copy()

    change_mapping = {'Ch': 1, 'No': 0}
    final2['change'] = final2['change'].map(change_mapping)

    # Mapping for 'diabetesMed' column
    diabetes_med_mapping = {'Yes': 1, 'No': 0}
    final2['diabetesMed'] = final2['diabetesMed'].map(diabetes_med_mapping)

    # Mapping for 'readmitted' column
    readmitted_mapping = {'>30': 2, 'NO': 1, '<30': 0}
    final2['readmitted'] = final2['readmitted'].map(readmitted_mapping)

    drug_columns = ['metformin', 'repaglinide', 'nateglinide', 'glimepiride', 'glipizide',
                'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin', 'glyburide-metformin']

    dosage_mapping = {'No': -1, 'Down': -2, 'Steady': 0, 'Up': 1}

    for col in drug_columns:
        final2[col] = final2[col].map(dosage_mapping)

    columns_drop=['discharge_disposition_id', 'admission_source_id','num_lab_procedures', 'num_procedures',
    'number_outpatient', 'number_emergency', 'number_inpatient',
                    'number_diagnoses','readmitted']
    final2.drop(columns=columns_drop,inplace=True)

    final2 = final2.drop_duplicates()
    return final2  # Adjust as needed

# Train KMeans and collaborative filtering models
def train_models(final2):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics.pairwise import cosine_similarity
    # Assuming you have a DataFrame named 'final_data'
    final_dataset = final2.copy()

    # Columns to consider for clustering
    cluster_columns = ['gender', 'age', 'weight', 'admission_type_id',
                    'time_in_hospital',
                    'num_medications',
                    'max_glu_serum', 'A1Cresult', 'change',
                    'diabetesMed', 'circulatory', 'diabetes', 'digestive',
                    'genitourinary', 'injuries', 'musculoskeletal', 'neoplasms',
                    'respiratory', 'other', 'race_Caucasian', 'race_AfricanAmerican',
                    'race_Other', 'race_Asian', 'race_Hispanic', 'metformin', 'repaglinide',
                    'nateglinide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
                    'rosiglitazone', 'insulin', 'glyburide-metformin']
    
    data_cluster = final_dataset[cluster_columns]

# Normalize and apply PCA
    scaler = StandardScaler()
    data_cluster_scaled = scaler.fit_transform(data_cluster)

    pca = PCA(n_components=34)  # You can adjust the number of components as needed
    data_cluster_pca = pca.fit_transform(data_cluster_scaled)

    # Perform KMeans clustering with 6 clusters
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(data_cluster_scaled)

    # Assign cluster labels to the original data
    final_dataset['cluster'] = cluster_labels

    drug_names=['metformin', 'repaglinide', 'nateglinide', 'glimepiride',
                         'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone',
                         'insulin', 'glyburide-metformin']

    train_data, test_data = train_test_split(final_dataset, test_size=0.2, random_state=42)
    # Calculate the collaborative matrix
    collaborative_matrix = np.zeros((5, len(drug_names)))

    for _, row in train_data.iterrows():
        cluster_idx = int(row['cluster'])
        drug_dosages = row[drug_names].values
        collaborative_matrix[cluster_idx] += drug_dosages

    # Normalize the collaborative matrix
    normalized_collaborative_matrix = collaborative_matrix / np.sum(collaborative_matrix, axis=1, keepdims=True)

    # Calculate cosine similarity for normalized collaborative matrix
    cosine_sim_matrix = cosine_similarity(normalized_collaborative_matrix)

    
    return kmeans, scaler, pca, normalized_collaborative_matrix, cosine_sim_matrix

# Save models to pickle files
def save_models(model, scaler, pca, collaborative_matrix, cosine_sim_matrix):
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
        
    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
        
    with open("pca.pkl", "wb") as pca_file:
        pickle.dump(pca, pca_file)
        
    with open("collaborative_matrix.pkl", "wb") as collaborative_matrix_file:
        pickle.dump(collaborative_matrix, collaborative_matrix_file)
        
    with open("cosine_sim_matrix.pkl", "wb") as cosine_sim_matrix_file:
        pickle.dump(cosine_sim_matrix, cosine_sim_matrix_file)

# Load models from pickle files
def load_models():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
        
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
        
    with open("pca.pkl", "rb") as pca_file:
        pca = pickle.load(pca_file)
        
    with open("collaborative_matrix.pkl", "rb") as collaborative_matrix_file:
        collaborative_matrix = pickle.load(collaborative_matrix_file)
        
    with open("cosine_sim_matrix.pkl", "rb") as cosine_sim_matrix_file:
        cosine_sim_matrix = pickle.load(cosine_sim_matrix_file)
    
        
    return model, scaler, pca, collaborative_matrix, cosine_sim_matrix

# Create and save pickle files
def create_pickle_files(data):
    final2 = preprocess_data(data)
    kmeans, scaler, pca, normalized_collaborative_matrix, cosine_sim_matrix = train_models(final2)
    save_models(kmeans, scaler, pca, normalized_collaborative_matrix, cosine_sim_matrix)

# Example usage
if __name__ == "__main__":
    main = pd.read_csv("diabetic_data_initial.csv")
    create_pickle_files(main)