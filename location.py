import warnings
warnings.filterwarnings("ignore")
import sys
import subprocess
import time
import numpy as np
import csitools
from passband import lowpass
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from read_pcap import NEXBeamformReader
from scipy.optimize import fsolve
from scipy.fftpack import fft, fftfreq

def run_command(command):
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command)

def countdown(seconds):
    for i in range(seconds, 0, -1):
        print(f"{i}...")
        time.sleep(1)

def smooth_data(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def smooth1_data(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def smooth2_data(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def derivative(data):
    return np.diff(data)

def find_main_frequency(signal, sampling_rate):
    fft_values = fft(signal)
    
    fft_frequencies = fftfreq(len(signal), 1 / sampling_rate)
    
    positive_freq_indices = np.where(fft_frequencies >= 0)
    fft_frequencies = fft_frequencies[positive_freq_indices]
    fft_values = np.abs(fft_values[positive_freq_indices])
    
    main_frequency = fft_frequencies[np.argmax(fft_values)]
    
    return main_frequency

def find_wave_indices(data,time):
    smoothed_data = smooth_data(data)
    first_derivative = derivative(smoothed_data)
    
    peak_index = np.argmin(data)

    mean_data = np.mean(data)
    
    start_index = peak_index
    damax = data[peak_index]
    #while data[start_index] >= 1.05*mean_data and start_index > 0:
    while data[start_index]<np.min(data[int(start_index-len(data)/10):int(start_index)]) and start_index > len(data)/10-1:
      while np.min(data[int(start_index-len(data)/15):start_index]) < 0.35*(np.mean(data[int(start_index-len(data)/15):start_index]) + damax) and start_index > len(data)/15-1:
        while start_index > len(data)/15-1 and first_derivative[start_index - 1] < 0:
              start_index -= 1
        start_index -= 1
      start_index -= 1
    dastart = data[start_index]

    end_index = peak_index
    while data[end_index] < 0.35*(dastart + damax) and end_index < len(first_derivative)-10 or np.min(data[end_index:int(end_index+len(data)/10)]) < 0.7*damax and end_index < len(first_derivative)-10:
      while end_index < len(first_derivative)-1 and first_derivative[end_index] > 0:
          end_index += 1
      end_index += 1

    '''while data[end_index]>np.min(data[int(end_index):int(end_index+len(data)/10)]) and end_index < len(data):
        while data[end_index] < 0.4*(dastart + damax) and end_index < len(first_derivative)-1 or np.min(data[end_index:int(end_index+len(data)/10)]) < 0.7*damax and end_index < len(first_derivative)-1:
            while end_index < len(first_derivative)-1 and first_derivative[end_index] > 0:
                end_index += 1
            end_index += 1
        end_index += 1'''

    start_var = np.var(data[int(start_index-len(data)/10):start_index])
    start_max = np.max(data[int(start_index-len(data)/10):start_index]) - np.min(data[int(start_index-len(data)/10):start_index])

    if end_index + len(data)/10 < len(data):
      end_var = np.var(data[end_index:int(end_index+len(data)/10)])
      end_max = np.max(data[end_index:int(end_index+len(data)/10)]) - np.min(data[end_index:int(end_index+len(data)/10)])
    else:
      end_var = np.var(data[end_index:len(data)])
      end_max = np.max(data[end_index:len(data)]) - np.min(data[end_index:len(data)])

    start_mean = np.mean(data[0:start_index])

    if end_index + start_index < len(data):
      end_mean = np.mean(data[len(data)-start_index:len(data)])
    else:
      end_mean = np.mean(data[len(data)-start_index:len(data)])
       
    if len(data)-end_index<1.5*len(data)/10 and end_index-start_index > 0.4*len(data):
       if np.var(data[int(len(data)/10):int(2*len(data)/5)])<1.2*np.var(data[int(3*len(data)/5):int(9*len(data)/10)]):
        return True, 0, 100
       else:
        return False, 0, 100

    if start_mean - end_mean > 35:
      if np.var(data[start_index:end_index])/0.5*(np.var(data[int(start_index-len(data)/10):start_index])+np.var(data[end_index:int(end_index+len(data)/10)]))<5:
          return False, 0, 100
      else:
          return False, time[start_index], time[end_index]
    elif end_mean - start_mean > 35:
      if np.var(data[start_index:end_index])/0.5*(np.var(data[int(start_index-len(data)/10):start_index])+np.var(data[end_index:int(end_index+len(data)/10)]))<5:
          return True, 0, 100
      else:
          return True, time[start_index], time[end_index]
    else:
      if start_var>end_var:
        if np.var(data[end_index:int(2*end_index-start_index)]) < 0:
          return True, 0, 100
        if np.var(data[start_index:end_index])/0.5*(np.var(data[int(start_index-len(data)/5):start_index])+np.var(data[end_index:int(end_index+len(data)/5)]))<5:
          return True, 0, 100
        else:
          return True, time[start_index], time[end_index]
      elif start_var>0.4*end_var and start_max>=0.85*end_max:
        if np.var(data[end_index:int(2*end_index-start_index)]) < 2:
          return True, 0, 100
        if np.var(data[start_index:end_index])/0.5*(np.var(data[int(start_index-len(data)/5):start_index])+np.var(data[end_index:int(end_index+len(data)/5)]))<5:
          return True, 0, 100
        else:
          return True, time[start_index], time[end_index]
      else:
        if np.var(data[start_index:end_index])/0.5*(np.var(data[int(start_index-len(data)/10):start_index])+np.var(data[end_index:int(end_index+len(data)/10)]))<5:
          return False, 0, 100
        else:
          return False, time[start_index], time[end_index]

def find_best_t(data):
    n = len(data)
    front_mean = np.mean(data[n * 0 // 5:n * 1 // 5])
    back_mean = np.mean(data[-n * 1 // 5:n])
    
    best_t = 0
    min_difference = float('inf')
    
    for t in range(1, n):
        constructed_data = np.concatenate([
            np.full(t, front_mean),
            np.full(n - t, back_mean)
        ])
        
        difference = np.linalg.norm(data[0:n] - constructed_data)
        
        if difference < min_difference:
            min_difference = difference
            best_t = t
    #print(t)

    return best_t, front_mean, back_mean

def calculate_transition_indices(data,time):
    data = smooth_data(data)
    n = len(data)
    t, front_mean, back_mean = find_best_t(data)
    if front_mean > back_mean:
        tf = t
        while data[tf] < 0.5*np.max(data[int(tf-len(data)/10):tf]) and tf > len(data)/10:
            while data[tf] < 0.9*front_mean:
                tf = tf-1
            tf = tf-1
        pre_t_point = tf
        if t < len(data):
          while data[t] > 0.9*back_mean and t<len(data):
            t = t+1
        post_t_point = t
    else:
        tf = t
        while data[tf] > 0.5*np.min(data[int(tf-len(data)/10):tf]) and tf > len(data)/10:
            while data[tf] > 0.9*front_mean:
                tf = tf-1
            tf = tf-1
        pre_t_point = tf
        while data[t] < 0.9*back_mean:
          t = t+1
        post_t_point = t
    if abs(np.mean(data[int(1*n/10):pre_t_point]) - np.mean(data[post_t_point:int(9*n/10)]))<6:
        return 0, 100
    elif time[post_t_point] - time[pre_t_point]>0.25:
      return time[pre_t_point], time[post_t_point]
    else:
      return time[pre_t_point], 0.5*(time[post_t_point]+time[pre_t_point])

def readpro_csi(filename):
    my_reader = NEXBeamformReader()
    csi_data = my_reader.read_file(filename,scaled=True)
    csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data)
    csi_matrix_first = csi_matrix[:, :, 0, 0]
    csi_matrix_first[csi_matrix_first == -np.inf] = np.nan
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    csi_matrix_first = imp_mean.fit_transform(csi_matrix_first)
    # Then we'll squeeze it to remove the singleton dimensions.
    csi_matrix_squeezed = np.squeeze(csi_matrix_first)
    csi_matrix_squeezed = np.transpose(csi_matrix_squeezed)
    for x in range(no_subcarriers-1):
        csi_matrix_squeezed[x] = lowpass(csi_matrix_squeezed[x], 3, 50, 5)
        #csi_matrix_squeezed[x] = hampel(csi_matrix_squeezed[x], 10, 3)
        #csi_matrix_squeezed[x] = running_mean(csi_matrix_squeezed[x], 10)

    #csi_matrix_squeezed = np.transpose(csi_matrix_squeezed)
    '''pca = PCA(n_components=3)
    csipca = pca.fit_transform(csi_matrix_squeezed)
    csipca = np.transpose(csipca)
    csipca0 = remove_data_with_high_variance(csipca[0])'''
    #csi_matrix_squeezed = np.transpose(csi_matrix_squeezed)
    row_means = np.mean(csi_matrix_squeezed, axis=1)
    top_5_indices = np.argsort(row_means)[-5:]
    top_5_rows = csi_matrix_squeezed[top_5_indices]
    sum_top_5_rows = np.sum(top_5_rows, axis=0)
    csi_mean = sum_top_5_rows - np.mean(sum_top_5_rows)
    csi_mean = remove_data_with_high_variance(csi_mean)
    x = csi_data.timestamps
    x = csi_data.timestamps - x[0]
    return x, csi_mean

def calculate_angle(time_ratio,frequency=2.4):
    distant = 3
    if frequency == 2.4:
        lambd = 0.125
    elif frequency == 5:
        lambd = 0.06
    body_len = 0.25

    def equation(angle):
        term1 = (8 * (2 * distant + lambd) * (2 * distant + lambd - 2 * distant * np.sin(angle))) / \
                ((2 * distant + lambd) ** 2 - (2 * distant * np.cos(angle)) ** 2)
        term2 = ((8 * distant + 4 * lambd - 8 * distant * np.sin(2)) * body_len) / \
                (lambd ** 2 + 4 * distant * lambd)
        return term1 + term2 - time_ratio

    initial_guess = 0.0  
    angle_solution = fsolve(equation, initial_guess)

    return np.degrees(angle_solution[0])

def remove_data_with_high_variance(data):
    data = np.array(data)
    
    for i in range(len(data) - 2, -1, -1):
        variance = np.var(data[i:])
        
        if variance > 2:
            return data[0:i]
            break

def location(file_name1 = r'capture_1.pcap', file_name2 = r'capture_2.pcap', file_name3 = r'capture_3.pcap', mac = 'None'):
    print(f"Start location the dangerous device {mac}")
    print(f"Processing the first captured path...")
    time1, csi_pca1 = readpro_csi(file_name1)
    ispos, start_index1,  end_index1 = find_wave_indices(csi_pca1,time1)
    print(f"Processing the second captured path...")
    time2,csi_pca2 = readpro_csi(file_name2)
    start_index2, end_index2 = calculate_transition_indices(csi_pca2,time2)
    time3,csi_pca3 = readpro_csi(file_name3)
    start_index3, end_index3 = calculate_transition_indices(csi_pca3,time3)
    sra = (np.max(csi_pca3[int(len(csi_pca3)/10):int(9*len(csi_pca3)/10)])-np.min(csi_pca3[int(len(csi_pca3)/10):int(9*len(csi_pca3)/10)]))/(np.max(csi_pca1[int(len(csi_pca1)/10):int(9*len(csi_pca1)/10)])-np.min(csi_pca1[int(len(csi_pca1)/10):int(9*len(csi_pca1)/10)]))
    device_angle = calculate_angle ((end_index1-start_index1)/(end_index2-start_index2))
    if end_index3 - start_index3 == 100:
       ispost = False
    elif sra < 0.6:
       ispost = False
    elif sra < 0.85 and np.var(csi_pca3[int(7*len(csi_pca3)/10):int(9*len(csi_pca3)/10)])>20:#normalization
       ispost = False
    else:
       ispost = True
    if ispost:
        print(f"\033[31m\033[47m\033[1mThe dangerous device is located at angle1: {device_angle}\033[0m")
    else:
        print(f"\033[31m\033[47m\033[1mThe dangerous device is located at angle: {180 - device_angle}\033[0m")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 location.py <mac> <channel>")
        sys.exit(1)

    mac = sys.argv[1]
    channel = sys.argv[2]

    # Run setup command
    setup_command = [
        'sudo', 'bash', 'setup.sh', '--laptop-ip', 'None', '--raspberry-ip', 'None',
        '--mac-adr', mac, '--channel', channel, '--bandwidth', '20', '--core', '1', '--spatial-stream', '1'
    ]
    run_command(setup_command)

    print("Dangerous device CSI monitoring setup complete. Press Enter to start the first collection.")
    input()
    print("Collection will start in 5 seconds. Please remain still.")
    countdown(5)
    
    # Run first tcpdump command
    tcpdump_command_1 = ['sudo', 'timeout', '10', 'tcpdump', '-i', 'wlan0', 'dst', 'port', '5500', '-w', 'capture_1.pcap']
    run_command(tcpdump_command_1)
    time.sleep(1)

    print("Press Enter to start the second collection.")
    input()
    print("Collection will start in 5 seconds. Please remain still.")
    countdown(5)

    # Run second tcpdump command
    tcpdump_command_2 = ['sudo', 'timeout', '10', 'tcpdump', '-i', 'wlan0', 'dst', 'port', '5500', '-w', 'capture_2.pcap']
    run_command(tcpdump_command_2)
    time.sleep(1)

    print("Press Enter to start the third collection.")
    input()
    print("Collection will start in 5 seconds. Please remain still.")
    countdown(5)
    # Run third tcpdump command
    tcpdump_command_2 = ['sudo', 'timeout', '10', 'tcpdump', '-i', 'wlan0', 'dst', 'port', '5500', '-w', 'capture_3.pcap']
    run_command(tcpdump_command_2)
    time.sleep(1)

    # Down mon0 for next wifi scan
    mondown_command = ['sudo', 'ip', 'link', 'set', 'mon0', 'down']
    run_command(mondown_command)
    time.sleep(1)

    location()
    time.sleep(1)

if __name__ == "__main__":
    main()

