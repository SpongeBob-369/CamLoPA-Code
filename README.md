# Implementation of CamLoPA
  **CamLoPA** is a framework for detecting and locating hidden wireless cameras, designed using a Raspberry Pi 4B and based on the analysis of wireless signal propagation paths. This repository contains the implementation code and steps for CamLoPA. Before proceeding, please ensure you have completed the following tasks:

- Install the nexmon-csi tool on your Raspberry Pi 4B. 
- Configure an external network card for wireless communication, to connect with your phone via an SSH tool (since nexmon-csi disables the wireless function, you will need to manually re-enable and configure it). 
- Set up another external network card with monitoring capabilities, ensuring it is set to monitor mode and named wlan2mon before use (the name can be modified in the code).

### Detection and Localization
Run **camscan.py** to automatically perform snooping camera detection and localization. During detection and localization, the user needs to follow the prompts and mimic the demonstration in the demo by walking for a total of 45 seconds.

**location.py** is the localization algorithm. As mentioned in Section VI C, CamLoPA scales the fluctuation durations of CSI for certain paths. For specific scaling parameters, please refer to this document.

Due to the difference in the RSSI values returned by the Raspberry Pi’s network card compared to standard methods, this code includes APs with readings of -39dBm or higher (as reported by the built-in Raspberry Pi network card) in the scanning range.

**Demo of CamLoPA**

Here is a demo video about CamLoPA: see CamLoPAdemo.mp4
[![Introduction Video](CamLoPAdemo.mp4)

Thanks for [nexmon_csi](https://github.com/seemoo-lab/nexmon_csi) and [CSIKit](https://github.com/Gi-z/CSIKit)

**Note**

The 360 camera can, in fact, be detected. In our paper, it was undetectable due to limitations of the external USB sniffer used. We recommend using the TX-N600, which is capable of capturing packets from the 360 camera. During detection, it's best to perform large-scale indoor movements to trigger traffic from the camera.
