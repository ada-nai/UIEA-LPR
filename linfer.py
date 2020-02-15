'''
Contains code for working with the Inference Engine.
You'll learn how to implement this code and more in
the related lesson on the topic.
'''

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        self.plugin = None
        self.input_blob = None
        self.exec_network = None


    def load_model(self, model, device="CPU", cpu_extension= '/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_avx2.so'):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        network = IENetwork(model=model_xml, weights=model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(network, device)

        # Get the input layer
        bl_iter = iter(network.inputs)
        self.input_blob_1 = next(bl_iter)
        self.input_blob_2 = next(bl_iter)
        
        # print('network input type: {}'.format(type(network.inputs)))
        # print('input blob type: {}'.format(type(self.input_blob)))
        # print('network inputs:{} {}'.format(self.input_blob_1, self.input_blob_2))
        # print('seq_ind:', network.inputs['seq_ind'])

        # Return the input shape (to determine preprocessing)
        return network.inputs[self.input_blob_1].shape


    def sync_inference(self, data, seq_ind):
        '''
        Makes a synchronous inference request, given an input image.
        '''
        #self.exec_network.infer({self.input_blob['data']:data, self.input_blob['seq_ind']:seq_ind})
        self.exec_network.infer({self.input_blob_1:data, self.input_blob_2:seq_ind})
        return  


    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[0].outputs
