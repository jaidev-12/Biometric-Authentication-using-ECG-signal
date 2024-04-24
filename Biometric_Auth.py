import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pywt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from fastdtw import fastdtw
from scipy.signal import find_peaks


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # Ensure that the length of the input vector is greater than the filter order
    if len(data) <= order:
        raise ValueError("The length of the input vector must be greater than the filter order.")
    
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Example usage:
# Replace 'your_ecg_data.csv' with the path to your CSV file and 'your_sampling_rate' with the sampling rate of your data.
csv_file_path = 'rec_30_2.csv'  # Replace this with the path to your CSV file
csv_file_path_y = 'rec_70_1.csv'
csv_file_path_y1 = 'rec_60_1.csv'
csv_file_path_y2 = 'rec_30_2.csv'
csv_file_path_y3 = 'rec_40_1.csv'
csv_file_path_y4 = 'rec_10_1.csv'
your_sampling_rate = 1000  # Replace this with your actual sampling rate

# Load ECG data from CSV
ecg_data = pd.read_csv(csv_file_path).iloc[0:,0].values  # Assuming 'ECG' is the column name in your CSV file
ecg_data_y = pd.read_csv(csv_file_path_y).iloc[0:,0].values
ecg_data_y1 = pd.read_csv(csv_file_path_y1).iloc[0:,0].values
ecg_data_y2 = pd.read_csv(csv_file_path_y2).iloc[0:,0].values
ecg_data_y3 = pd.read_csv(csv_file_path_y3).iloc[0:,0].values
ecg_data_y4 = pd.read_csv(csv_file_path_y4).iloc[0:,0].values



lowcut = 2.0  # Lower cutoff frequency in Hz
highcut = 50.0  # Upper cutoff frequency in Hz

filtered_ecg = bandpass_filter(ecg_data, lowcut, highcut, your_sampling_rate)
filtered_ecg_y = bandpass_filter(ecg_data_y, lowcut, highcut, your_sampling_rate)
filtered_ecg_y1 = bandpass_filter(ecg_data_y1, lowcut, highcut, your_sampling_rate)
filtered_ecg_y2 = bandpass_filter(ecg_data_y2, lowcut, highcut, your_sampling_rate)
filtered_ecg_y3 = bandpass_filter(ecg_data_y3, lowcut, highcut, your_sampling_rate)
filtered_ecg_y4 = bandpass_filter(ecg_data_y4, lowcut, highcut, your_sampling_rate)

ecg_csv = [ecg_data, ecg_data_y,ecg_data_y1,ecg_data_y2,ecg_data_y3,ecg_data_y4]
filtered_signals = [filtered_ecg, filtered_ecg_y, filtered_ecg_y1, filtered_ecg_y2,filtered_ecg_y3,filtered_ecg_y4]

for i in range(len(ecg_csv)):
    # Plot the original and filtered signals in separate subplots
    plt.figure(figsize=(12, 8))

    # Plot original ECG signal
    plt.subplot(2, 1, 1)
    plt.plot(ecg_csv[i], label='Original ECG')
    plt.title('Original ECG Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot filtered ECG signal
    plt.subplot(2, 1, 2)
    plt.plot(filtered_signals[i], label='Filtered ECG (2-50 Hz)')
    plt.title('Filtered ECG Signal (2-50 Hz)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()



def detect_all_peaks(ecg_signal, threshold_factor=0.62):
    # Find peaks in the ECG signal using adaptive thresholding
    peaks, _ = find_peaks(ecg_signal, height=np.max(ecg_signal) * threshold_factor)

    return peaks

# Example usage:
# Replace 'filtered_ecg_signal' with your actual filtered ECG signal

for i in range(len(ecg_csv)):
    detected_peaks = detect_all_peaks((filtered_signals[i])[:3000])

    # Plot the ECG signal with all detected peaks
    plt.figure(figsize=(14, 5))
    plt.plot(filtered_signals[i][:3000], label='Filtered ECG Signal')
    plt.plot(detected_peaks, filtered_signals[i][:3000][detected_peaks], 'go', label='Detected Peaks')
    plt.title('Detection of All Peaks in Filtered ECG Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


# Simulate an ECG signal (replace this with your actual signal)
fs = 1000  # Sampling frequency



def wavelet_feature_extraction(ecg_signal, wavelet='sym4', level=4):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)

    # Plot the original signal and wavelet coefficients
    plt.figure(figsize=(13, 10))
    plt.plot(ecg_signal)
    plt.title('Original ECG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    
        
    plt.figure(figsize=(20, 18))
    plt.subplot(6,1,1)
    plt.plot(coeffs[0])
    plt.title(f'Level {1} Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Amplitude')
    
    plt.subplot(6,1,3)
    plt.plot(coeffs[1])
    plt.title(f'Level {2} Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Amplitude')
    
    plt.subplot(6,1,6)
    plt.plot(coeffs[2])
    plt.title(f'Level {3} Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Amplitude')
    
    plt.figure(figsize=(20, 18))
    plt.subplot(5,1,1)
    plt.plot(coeffs[3])
    plt.title(f'Level {4} Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Amplitude')
    
    plt.subplot(5,1,3)
    plt.plot(coeffs[4])
    plt.title(f'Level {5} Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Amplitude')
    
    plt.show()

    # Extract statistical features from wavelet coefficients
    feature_vector = np.hstack([np.mean(c) for c in coeffs])

    # Display the extracted features
    print("Extracted Features:")
    print(feature_vector)

    return feature_vector

# Example usage:
# Replace ecg_signal with your actual ECG signal
 # Replace this with your actual ECG signal
feature_vector = wavelet_feature_extraction(filtered_ecg)
feature_vector_y = wavelet_feature_extraction(filtered_ecg_y)
feature_vector_y1 = wavelet_feature_extraction(filtered_ecg_y1)
feature_vector_y2 = wavelet_feature_extraction(filtered_ecg_y2)
feature_vector_y3 = wavelet_feature_extraction(filtered_ecg_y3)
feature_vector_y4 = wavelet_feature_extraction(filtered_ecg_y4)


feature_vector_array = [feature_vector, feature_vector_y, feature_vector_y1, feature_vector_y2, feature_vector_y3, feature_vector_y4]




def calculate_dtw_distance(template_feature_vector, query_feature_vector):
    distance, path = fastdtw(template_feature_vector, query_feature_vector)
    return distance, path

# Example usage:
# Replace template_feature_vector and query_feature_vector with your actual feature vectors
template_feature_vector = feature_vector
query_feature_vector = filtered_signals
dtw_distance_array = []
alignment_path_array = []
# Calculate DTW distance and path
for i in range(1,len(ecg_csv)):
    dtw_distance, alignment_path = calculate_dtw_distance(template_feature_vector, query_feature_vector[i])
    print(f"DTW Distance: {dtw_distance}")
    dtw_distance_array.append(dtw_distance)
    alignment_path_array.append(alignment_path)

# Plot the alignment matrix heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(np.array(alignment_path).T, origin='lower', cmap='viridis', interpolation='nearest')
    plt.title('DTW Alignment Matrix')
    plt.xlabel('Template Feature Vector Index')
    plt.ylabel('Query Feature Vector Index')
    plt.colorbar(label='Alignment Cost')
    plt.show()

# Display the DTW distance



genuine_scores = feature_vector
imposter_scores = feature_vector_array



def calculate_roc(genuine_scores, imposter_scores):
    # Combine genuine and imposter scores and labels
    all_scores = np.concatenate([genuine_scores, imposter_scores])
    true_labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])

    # Compute ROC curve and area under the curve (AUC)
    fpr, tpr, thresholds = roc_curve(true_labels, -all_scores)  # Using -scores because roc_curve considers higher values as better
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Find the threshold that gives a balanced error rate (equal FPR and FNR)
    eer_threshold = thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]
    print(f"Equal Error Rate (EER) Threshold: {eer_threshold}")

    return eer_threshold


threshold_array = []
# Example usage:
# Replace genuine_scores and imposter_scores with your actual score arrays
for i in range(1, len(ecg_csv)):
    threshold = calculate_roc(genuine_scores, imposter_scores[i])
    print(f"Chosen Threshold{i}: {threshold}")
    threshold_array.append(threshold)
    






def authenticate_claimed_identity(XTI, Xq, threshold, names):
    distance = calculate_dtw_distance(XTI, Xq)[0]
    
    print("Distance", distance)
    
    if distance <= threshold:
        print(f"{names} Identity accepted as a genuine user.")
    else:
        print(f"{names} Identity rejected. Considered an imposter.")

# Example usage:
# Replace these with your actual feature vectors and threshold
XTI = feature_vector
Xq = feature_vector_array
names = ['ecg0', 'ecg1', 'ecg2', 'ecg3', 'ecg4', 'ecg5']


for i in range(1, len(ecg_csv)):
    # Authenticate the claimed identity
    authenticate_claimed_identity(XTI, Xq[i], threshold,names[i])




def evaluate_metrics(genuine_scores, imposter_scores):
    # Combine genuine and imposter scores
    all_scores = np.concatenate((genuine_scores, imposter_scores))
    
    # Create labels for genuine (1) and imposter (0) scores
    labels = np.concatenate((np.ones_like(genuine_scores), np.zeros_like(imposter_scores)))
    
    # Calculate ROC curve and area under the curve (AUC)
    fpr, tpr, _ = roc_curve(labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate equal error rate (EER)
    eer = np.abs(fpr[np.argmax(tpr) - 1] - (1 - tpr[np.argmax(tpr) - 1]))
    
    # Calculate false accept rate (FAR), false reject rate (FRR), and accuracy
    distance = (calculate_dtw_distance(XTI, imposter_scores))[0]
    far = fpr[np.argmax(tpr) - 1]
    if distance <= threshold:
        frr = 0
    else:
        frr = 1 - tpr[np.argmax(tpr) - 1]
    
    
    return roc_auc, eer, far, frr

print("Imposter_scores ",imposter_scores)

#for i in range(1, len(ecg_csv)):
    #roc_auc, eer, far, frr = evaluate_metrics(genuine_scores, imposter_scores[i])
    #print(f"Resut for ECG Signal {i} \n")
    # Display the results
    #print("ROC AUC:", roc_auc)
    #print("Equal Error Rate (EER):", eer)
    #print("False Accept Rate (FAR):", far)
    #print("False Reject Rate (FRR):", frr)
    #print("\n")

#def calculate_accuracy(decisions, ground_truth):
    # Ensure decisions and ground_truth have the same length
    #if len(decisions) != len(ground_truth):
        #raise ValueError("Decisions and ground_truth must have the same length.")

    # Calculate accuracy
    #correct_predictions = np.sum(decisions == ground_truth)
    #total_samples = len(ground_truth)
    #accuracy = (correct_predictions / total_samples)*2

    #return accuracy

# Example usage:
# Replace decisions and ground_truth with your actual decisions and ground truth

#for i in range(1, len(ecg_csv)):
    #ground_truth = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores[i])])
    #all_scores = np.concatenate([genuine_scores, imposter_scores[i]])
    #decisions = (all_scores >= threshold).astype(int)  # Using the threshold determined from ROC analysis

    #accuracy = calculate_accuracy(decisions, ground_truth)
    #print(f"Accuracy for ECG Signal {i}:", accuracy*100, "%")

