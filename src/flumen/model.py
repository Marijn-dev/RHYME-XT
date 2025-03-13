import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class CausalFlowModel(nn.Module):

    def __init__(self,
                 state_dim,
                 control_dim,
                 output_dim,
                 control_rnn_size,
                 control_rnn_depth,
                 encoder_size,
                 encoder_depth,
                 decoder_size,
                 decoder_depth,
                 use_POD,
                 use_trunk,
                 use_petrov_galerkin,
                 use_fourier,
                 use_conv_encoder,
                 trunk_size,
                 POD_modes,
                 trunk_modes,
                 fourier_modes,
                 use_batch_norm):
        super(CausalFlowModel, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        self.control_rnn_size = control_rnn_size
        self.basis_function_modes = self.state_dim

        self.POD_enabled = use_POD
        self.POD_modes = POD_modes
        
        self.trunk_enabled = use_trunk
        self.trunk_modes = trunk_modes
        self.trunk_size = trunk_size

        self.fourier_enabled = use_fourier
        self.fourier_modes = fourier_modes

        self.conv_encoder_enabled = use_conv_encoder

        self.petrov_galerkin_enabled = use_petrov_galerkin # whether to use galerkin or standard galerkin
        self.projection = (self.POD_enabled or self.trunk_enabled) and self.petrov_galerkin_enabled == False

        if self.POD_enabled:
            self.POD_modes = min(self.POD_modes,self.state_dim) # cant be higher than state dimension
            self.basis_function_modes = self.POD_modes  
            self.out_size_decoder = self.basis_function_modes

            # petrov galerkin -> use different basis functions for branch and trunk
            if self.petrov_galerkin_enabled:
                self.in_size_encoder = self.control_dim = self.state_dim
            
            # standard galerkin -> same basis functions for branch and trunk
            else:
                self.in_size_encoder = self.control_dim = self.basis_function_modes

        elif self.trunk_enabled:
            self.basis_function_modes = self.trunk_modes  
            self.out_size_decoder = self.basis_function_modes

            # petrov galerkin -> use different basis functions for branch and trunk
            if self.petrov_galerkin_enabled:
                self.in_size_encoder = self.control_dim = self.state_dim

            # standard galerkin -> same basis functions for branch and trunk
            else:
                self.in_size_encoder = self.control_dim = self.basis_function_modes
                

        else:
            self.in_size_encoder = self.control_dim = self.state_dim
            self.out_size_decoder = self.output_dim
            

        if self.fourier_enabled:
            self.fourier_modes = min(self.fourier_modes, self.basis_function_modes // 2)
            self.in_size_encoder = self.control_dim = self.fourier_modes * 2


        self.u_rnn = torch.nn.LSTM(
            input_size=1 + self.control_dim,
            hidden_size=control_rnn_size,
            batch_first=True,
            num_layers=control_rnn_depth,
            dropout=0,
        )

        x_dnn_osz = control_rnn_depth * control_rnn_size

        # Flow encoder (CNN)
        if self.conv_encoder_enabled:
            self.x_dnn = DynamicPoolingCNN(in_size=self.in_size_encoder,
                           out_size=x_dnn_osz,use_batch_norm=use_batch_norm)
            
        # Flow encoder (MLP)
        else:
            self.x_dnn = FFNet(in_size=self.in_size_encoder,
                           out_size=x_dnn_osz,
                           hidden_size=encoder_depth *
                           (encoder_size * x_dnn_osz, ), 
                           use_batch_norm=use_batch_norm)

        # Flow decoder
        u_dnn_isz = control_rnn_size
        self.u_dnn = FFNet(in_size=u_dnn_isz,
                           out_size=self.out_size_decoder,
                           hidden_size=decoder_depth *
                           (decoder_size * u_dnn_isz, ),
                           use_batch_norm=use_batch_norm)
        
        # Trunk network
        if self.trunk_enabled:
            self.trunk = TrunkNet(in_size=1,
                            out_size=self.basis_function_modes,
                            hidden_size=self.trunk_size,
                            use_batch_norm=use_batch_norm)
            


    def forward(self, x, rnn_input,PHI,locations, deltas):
        unpadded_u, unpacked_lengths = pad_packed_sequence(rnn_input, batch_first=True) # unpack input
        u = unpadded_u[:, :, :-1]                                         # extract inputs values

        basis_functions = 0

        # POD basis functions
        if self.POD_enabled:
            basis_functions += PHI[:, :self.basis_function_modes]

        # Trunk MLP basis functions
        if self.trunk_enabled:
            basis_functions +=  self.trunk(locations.view(-1, 1))  

        # if normal galerking -> project the inputs
        if self.projection:
            x = torch.einsum("ni,bn->bi",basis_functions,x) 
            u = torch.einsum('ni,btn->bti',basis_functions,u) 

        if self.fourier_enabled:
            x_fft = torch.fft.rfft(x) 
            x_fft = x_fft[:,:self.fourier_modes] # retain only self.fourier_modes
            x = torch.cat([x_fft.real, x_fft.imag],dim=-1) # concatenate real and imag coefficients

            u_fft = torch.fft.rfft(u, dim=-1)  
            u_fft = u_fft[:,:,:self.fourier_modes]
            u = torch.cat([u_fft.real, u_fft.imag], dim=-1) 
             
        # repack input
        u_deltas = torch.cat((u, deltas), dim=-1)          
        rnn_input = pack_padded_sequence(u_deltas, unpacked_lengths, batch_first=True)


        h0 = self.x_dnn(x)
        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.zeros_like(h0)

        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0, c0))
        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed,
                                                           batch_first=True)
        h_shift = torch.roll(h, shifts=1, dims=1)
        h_shift[:, 0, :] = h0[-1]
        encoded_controls = (1 - deltas) * h_shift + deltas * h
     
        output_flow = self.u_dnn(encoded_controls[range(encoded_controls.shape[0]),
                                             h_lens - 1, :])
        
        # Regular
        if self.POD_enabled == False and self.trunk_enabled == False:
            output = output_flow
        
        # Inner product
        else:
            output = torch.einsum("ni,bi->bn",basis_functions,output_flow)

        return output

## MLP
class FFNet(nn.Module):

    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 use_batch_norm,
                 activation=nn.Tanh):
        super(FFNet, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size[0]))

        if use_batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_size[0]))

        self.layers.append(activation())

        for isz, osz in zip(hidden_size[:-1], hidden_size[1:]):
            self.layers.append(nn.Linear(isz, osz))

            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(osz))

            self.layers.append(activation())
            self.layers.append(nn.Dropout(0.2))  # Dropout layer

        self.layers.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
    
class TrunkNet(nn.Module):

    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 use_batch_norm,
                 activation=nn.Tanh):
        super(TrunkNet, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size[0]))
        self.layers.append(activation())
        
        if use_batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_size[0]))

        for isz, osz in zip(hidden_size[:-1], hidden_size[1:]):
            self.layers.append(nn.Linear(isz, osz))
            self.layers.append(activation())
            self.layers.append(nn.Dropout(0.2))  # Dropout layer

            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(osz))


        self.layers.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input


class DynamicPoolingCNN(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 use_batch_norm,
                 activation=nn.ReLU):
        super(DynamicPoolingCNN, self).__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.input_dim = in_size
        self.output_dim = out_size
        self.output_len = 50

        self.layers = nn.ModuleList()

        conv_channels = [1,16,32,64,128]
        for isz, osz in zip(conv_channels[:-1], conv_channels[1:]):
            self.layers.append(nn.Conv1d(in_channels=isz, 
                                         out_channels=osz, 
                                         kernel_size=5, 
                                         stride=1, 
                                         padding=2))
            
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(osz))

            self.layers.append(activation())    

        self.layers.append(self.dropout)

        # Fully connected layer for transforming to output size
        self.fc = nn.Linear(conv_channels[-1] * self.output_len, self.output_dim)
    
    def calculate_pooling_params(self, input_length, output_len):
        """ Calculate the kernel_size based on input size and desired output length """
        
        # Calculate kernel size based on input length and output length
        kernel_size = max(1, input_length // output_len)  # Ensure kernel size is at least 1
        return kernel_size
    
    def forward(self, input):

        # Reshape input to (batch_size, 1, input_dim) for CNN
        input = input.unsqueeze(1)  # Assuming input has shape (batch_size, input_dim)
        input_dim = input.shape[2]

        # convolutional layers
        for layer in self.layers:
            input = layer(input)
        
        # Apply dynamic pooling 
        self.kernel_size = self.calculate_pooling_params(input_dim, self.output_len)
        pool = nn.AvgPool1d(kernel_size=self.kernel_size)
        input = pool(input)
        
        # Flatten the output from pooling layer and pass through fully connected layer
        input = input.view(input.size(0), -1)  # Flatten the tensor
        input = self.fc(input)
        
        return input