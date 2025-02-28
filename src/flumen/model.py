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
                 use_fourier,
                 use_conv_encoder,
                 trunk_size,
                 POD_modes,
                 fourier_modes,
                 use_batch_norm=False):
        super(CausalFlowModel, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        
        self.control_rnn_size = control_rnn_size

        self.POD_enabled = use_POD
        self.trunk_enabled = use_trunk
        self.conv_encoder_enabled = use_conv_encoder
        self.POD_modes = POD_modes
        self.fourier_modes = fourier_modes
        self.trunk_size = trunk_size
        
        self.fourier_enabled = use_fourier
        if self.POD_enabled:
            assert self.POD_modes <= self.state_dim,  'POD_modes too high'
        assert self.fourier_modes <= self.state_dim // 2,  'fourier_modes too high'

        if self.POD_enabled:
            self.in_size_encoder,self.out_size_decoder,self.control_dim = self.POD_modes,self.POD_modes,self.POD_modes


        elif self.trunk_enabled:
            if self.fourier_enabled:
                self.in_size_encoder = self.fourier_modes * 2
                self.control_dim = self.fourier_modes * 2

            else:
                self.in_size_encoder = state_dim

            self.out_size_decoder = self.POD_modes

        # Fourier Net
        elif self.fourier_enabled:
            self.in_size_encoder = self.fourier_modes * 2
            self.out_size_decoder = output_dim
            self.control_dim = self.fourier_modes * 2

        # regular flow
        else:
            self.in_size_encoder = state_dim
            self.out_size_decoder = output_dim

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
                           out_size=x_dnn_osz)
            
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
                            out_size=self.POD_modes,
                            hidden_size=self.trunk_size,
                            use_batch_norm=use_batch_norm)
            

    def forward(self, x, rnn_input,PHI,locations, deltas):
        if self.fourier_enabled:
            x_fft = torch.fft.rfft(x) 
            x_fft = x_fft[:,:self.fourier_modes] # retain only self.fourier_modes
            x0 = torch.cat([x_fft.real, x_fft.imag],dim=-1) # concatenate real and imag

            unpadded_u, unpacked_lengths = pad_packed_sequence(rnn_input, batch_first=True) # unpack input
            deltas = unpadded_u[:, :, -1:]   
            u = unpadded_u[:, :, :-1]                                         # extract inputs values

            u_fft = torch.fft.rfft(u, dim=-1)  
            u_fft = u_fft[:,:,:self.fourier_modes]
            u_fft = torch.cat([u_fft.real, u_fft.imag], dim=-1) 
             
            u_deltas = torch.cat((u_fft, deltas), dim=-1)  
            input = pack_padded_sequence(u_deltas, unpacked_lengths, batch_first=True)
        
        # Project inputs to Flow model
        elif self.POD_enabled: 
            x0 = torch.einsum("ni,bn->bi",PHI[:,:self.POD_modes],x) 

            unpadded_u, unpacked_lengths = pad_packed_sequence(rnn_input, batch_first=True) # unpack input
            deltas = unpadded_u[:, :, -1:]                                    # extract deltas
            u = unpadded_u[:, :, :-1]                                         # extract inputs values
            
            u_projected = torch.einsum('ni,btn->bti',PHI[:,:self.POD_modes],u) # project inputs
            u_deltas = torch.cat((u_projected, deltas), dim=-1)  
            input = pack_padded_sequence(u_deltas, unpacked_lengths, batch_first=True)

        else:
            x0 = x
            input = rnn_input


        h0 = self.x_dnn(x0)
        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.zeros_like(h0)

        rnn_out_seq_packed, _ = self.u_rnn(input, (h0, c0))
        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed,
                                                           batch_first=True)

        h_shift = torch.roll(h, shifts=1, dims=1)
        h_shift[:, 0, :] = h0[-1]

        encoded_controls = (1 - deltas) * h_shift + deltas * h
        output_flow = self.u_dnn(encoded_controls[range(encoded_controls.shape[0]),
                                             h_lens - 1, :])
        
        # Regular
        if self.POD_enabled == False and self.trunk_enabled == False:
            return output_flow
            

        elif self.POD_enabled:
            ## POD + Trunk
            if self.trunk_enabled:
                trunk_output =  self.trunk(locations.view(-1,1))
                basis_functions = PHI[:,:self.POD_modes] + trunk_output 
                
            ## POD
            else:
                basis_functions = PHI[:,:self.POD_modes]

        ## Trunk
        else:
            basis_functions = self.trunk(locations.view(-1,1))

        output = torch.einsum("ni,bi->bn",basis_functions,output_flow)
        return output

## MLP
class FFNet(nn.Module):

    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 activation=nn.Tanh,
                 use_batch_norm=False):
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
                 activation=nn.Tanh,
                 use_batch_norm=False):
        super(TrunkNet, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size[0]))
        self.layers.append(activation())

        for isz, osz in zip(hidden_size[:-1], hidden_size[1:]):
            self.layers.append(nn.Linear(isz, osz))
            self.layers.append(activation())
            self.layers.append(nn.Dropout(0.2))  # Dropout layer


        self.layers.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

class CONV_Encoder(nn.Module):
    def __init__(self,
                 in_size,
                 out_size):         
        super(CONV_Encoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.relu = nn.ReLU() 

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        
        # Windowed Average Pooling (WAP)
        self.pool_wap1 = nn.AvgPool1d(kernel_size=5, stride=2, padding=2)  # Reduces length by 2x
        self.pool_wap2 = nn.AvgPool1d(kernel_size=5, stride=2, padding=2)  # Further reduction
        
        # Final Global Average Pooling (All-AP)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Collapses spatial dimension

        # Fully connected layer, get correct dimension
        self.fc = nn.Linear(256, out_size)

    def forward(self, input):

        input = input.unsqueeze(1)  # (batch_size, input_dim) -> (batch_size, 1, input_dim)

        input = self.relu(self.conv1(input))
        input = self.pool_wap1(input)  

        input = self.relu(self.conv2(input))
        input = self.pool_wap2(input)

        input = self.relu(self.conv3(input))

        input = self.global_pool(input)  # Output shape: (batch_size, 256, 1)
        
        input = input.view(input.size(0), -1)  # (batch_size, 256)
        output = self.fc(input)
        
        return output


class DynamicPoolingCNN(nn.Module):
    def __init__(self,
                 in_size,
                 out_size):
        super(DynamicPoolingCNN, self).__init__()

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.input_dim = in_size
        self.output_dim = out_size
        self.output_len = 50

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        
        # Fully connected layer for transforming to output size
        self.fc = nn.Linear(128 * self.output_len, self.output_dim)
    
    def calculate_pooling_params(self, input_length, output_len):
        """ Calculate the kernel_size based on input size and desired output length """
        
        # Calculate kernel size based on input length and output length
        kernel_size = input_length // output_len
        
        # Ensure kernel size is at least 1
        if kernel_size < 1:
            kernel_size = 1
            stride = 1
        
        return kernel_size
    
    def forward(self, input):

        # Reshape input to (batch_size, 1, input_dim) for CNN
        input = input.unsqueeze(1)  # Assuming input has shape (batch_size, input_dim)
        input_dim = input.shape[2]

        # Apply convolutions
        input = self.activation(self.conv1(input))
        input = self.activation(self.conv2(input))
        input = self.activation(self.conv3(input))
        input = self.dropout(input)
        
        # Apply dynamic pooling 
        self.kernel_size = self.calculate_pooling_params(input_dim, self.output_len)
        pool = nn.AvgPool1d(kernel_size=self.kernel_size)
        input = pool(input)
        
        # Flatten the output from pooling layer and pass through fully connected layer
        input = input.view(input.size(0), -1)  # Flatten the tensor
        output = self.fc(input)
        
        return output