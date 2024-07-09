import socket
import subprocess
import dpkt
import struct

HOST = "192.168.31.28"
PORT = 65433

# Define the tcpdump command
command = ['sudo', 'tcpdump', '-i', 'wlan0', 'dst', 'port', '5500',  '-vv',  '-c', '1000000', '--immediate-mode', '-w', '-']
# Start the tcpdump process and capture its output
process = subprocess.Popen(command, stdout=subprocess.PIPE)

# Create a TCP socket and connect to the receiver
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (HOST, PORT)
sock.connect(server_address)

output_file = 'sender_packets.pcap'
output_pcap = open(output_file, 'wb')
pcap_writer = dpkt.pcap.Writer(output_pcap)

# Loop through the packets in the output and send them over the socket
i = 0
for ts, msg_bytes in dpkt.pcap.Reader(process.stdout):
    msg_len = len(msg_bytes)
    len_bytes = struct.pack('>I', msg_len)
    prefixed_msg = len_bytes + msg_bytes
    print(i, msg_len)
    sock.sendall(prefixed_msg)
    pkt = dpkt.ethernet.Ethernet(msg_bytes)
    pcap_writer.writepkt(pkt)
    i += 1
# Close the socket
sock.close()
