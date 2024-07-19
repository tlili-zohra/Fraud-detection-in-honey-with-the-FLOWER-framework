Open Anaconda Prompt. Run python server.py (in strategie : min_available_clients=5)
open 5 windows of Anaconda Prompt
Run handomly : 
Wind1 : python client.py --parition-id 0
Wind2 : python client.py --parition-id 1
Wind3 : python client.py --parition-id 2
Wind4 : python client.py --parition-id 3
Wind5 : python client.py --parition-id 4
NB. the important partition-id never exceeds in federatedDataset the partition of dataset.
