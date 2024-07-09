import csitools
import numpy as np
from passband import lowpass
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from read_pcap import NEXBeamformReader


def smooth_data(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# 计算一阶导数
def derivative(data):
    return np.diff(data)

def find_wave_indices(data):
    smoothed_data = smooth_data(data)
    first_derivative = derivative(smoothed_data)
    
    # 找到最高波峰的索引
    peak_index = np.argmax(smoothed_data)
    
    # 从波峰向前搜索波的开始索引
    start_index = peak_index
    while start_index > 0 and first_derivative[start_index - 1] > 0:
        start_index -= 1
    
    # 从波峰向后搜索波的结束索引
    end_index = peak_index
    while end_index < len(first_derivative) and first_derivative[end_index] < 0:
        end_index += 1
    
    return start_index, peak_index, end_index


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def moving_variance(data, window_size):
    mean = moving_average(data, window_size)
    return moving_average((data[window_size-1:] - mean) ** 2, window_size)

def detect_transition(data, window_size=20, threshold=50):
    mean = moving_average(data, window_size)
    variance = moving_variance(data, window_size)
    
    transition_start = None
    transition_end = None

    for i in range(len(variance)):
        if variance[i] > threshold and transition_start is None:
            transition_start = i
        if variance[i] < threshold and transition_start is not None:
            transition_end = i
            break

    return transition_start, transition_end

datapath = r'capture_1.pcap'
my_reader = NEXBeamformReader()
csi_data = my_reader.read_file(datapath,scaled=True)
csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data)

csi_matrix_first = csi_matrix[:, :, 0, 0]
# Then we'll squeeze it to remove the singleton dimensions.
csi_matrix_squeezed = np.squeeze(csi_matrix_first)

csi_matrix_squeezed = np.transpose(csi_matrix_squeezed)
for x in range(no_subcarriers):
  csi_matrix_squeezed[x] = lowpass(csi_matrix_squeezed[x], 3, 50, 5)
  #csi_matrix_squeezed[x] = hampel(csi_matrix_squeezed[x], 10, 3)
  #csi_matrix_squeezed[x] = running_mean(csi_matrix_squeezed[x], 10)

csi_matrix_squeezed = np.transpose(csi_matrix_squeezed)
pca = PCA(n_components=3)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
csiimp = imp_mean.fit_transform(csi_matrix_squeezed)
csipca = pca.fit_transform(csiimp)
csipca = np.transpose(csipca)
x = csi_data.timestamps
x = csi_data.timestamps - x[0]
start_index, peak_index, end_index = find_wave_indices(csipca[0])
print(f"Start Index: {x[start_index]}, Peak Index: {peak_index}, End Index: {x[end_index]}")
start_index, end_index = detect_transition(csipca[0])
print(f"Start Index: {x[start_index + 20 - 1]}, End Index: {x[end_index + 20 - 1]}")
plt.plot(csipca[0])
plt.show()
'''csi_matrix_squeezed = np.transpose(csi_matrix_squeezed)
BatchGraph.plot_heatmap(csi_matrix_squeezed, csi_data.timestamps)'''