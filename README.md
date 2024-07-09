# Implementation of CamLoPA
  **CamLoPA** is a framework for detecting and locating hidden wireless cameras, designed using a Raspberry Pi 4B and based on the analysis of wireless signal propagation paths. This repository contains the implementation code and steps for CamLoPA. Before proceeding, please ensure you have completed the following tasks:

- Install the nexmon-csi tool on your Raspberry Pi 4B. 
- Configure an external network card for wireless communication, to connect with your phone via an SSH tool (since nexmon-csi disables the wireless function, you will need to manually re-enable and configure it). 
- Set up another external network card with monitoring capabilities, ensuring it is set to monitor mode and named wlan2mon before use (the name can be modified in the code).

## Detection and Localization
Run camscan.py to automatically perform suspicious device detection and localization. During detection and localization, the user needs to follow the prompts and mimic the demonstration in the demo by walking for a total of 45 seconds.
