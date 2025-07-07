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
                 use_nonlinear,
                 IC_encoder_decoder,
                 regular,
                 use_conv_encoder,
                 trunk_size_svd,
                 trunk_size_extra,
                 NL_size,
                 trunk_modes,
                 trunk_model,
                 use_batch_norm):
        super(CausalFlowModel, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        self.control_rnn_size = control_rnn_size
        self.trunk_modes = trunk_modes
        self.trunk_size_svd = trunk_size_svd
        self.trunk_size_extra = trunk_size_extra
        self.NL_size = NL_size
        self.conv_encoder_enabled = use_conv_encoder
        self.nonlinear_enabled = use_nonlinear 
        self.IC_encoder_decoder_enabled = IC_encoder_decoder
        self.regular_enabled = regular
        self.basis_function_modes = self.trunk_modes  
        self.out_size_decoder = self.basis_function_modes
        self.in_size_encoder = self.control_dim = self.basis_function_modes
        self.control_rnn_depth = control_rnn_depth

        self.u_rnn = torch.nn.RNN(
            input_size=1 + self.control_dim,
            hidden_size=control_rnn_size,
            batch_first=True,
            num_layers=control_rnn_depth,
            dropout=0,
        )

        ### Enforce IC ###
        if self.IC_encoder_decoder_enabled:
            assert control_rnn_size > self.trunk_modes, "Control RNN size must be greater than trunk modes"	
            x_dnn_osz = control_rnn_depth * (control_rnn_size-self.trunk_modes)
        else: 
            x_dnn_osz = control_rnn_depth * control_rnn_size
            
            ### Flow decoder (MLP) ###
            self.u_dnn = FFNet(in_size=control_rnn_size,
                            out_size=self.out_size_decoder,
                            hidden_size=decoder_depth *
                            (decoder_size * control_rnn_size, ),
                            use_batch_norm=use_batch_norm)
        
        ### Flow encoder (CNN) ###
        if self.conv_encoder_enabled:
            self.x_dnn = CNN_encoder(in_size=self.in_size_encoder,
                           out_size=x_dnn_osz,use_batch_norm=use_batch_norm)
            
        ### Flow encoder (MLP) ###
        else:
            self.x_dnn = FFNet(in_size=self.in_size_encoder,
                           out_size=x_dnn_osz,
                           hidden_size=encoder_depth *
                           (encoder_size * x_dnn_osz, ), 
                           use_batch_norm=use_batch_norm)

        ### Trunk (MLP) ###
        self.trunk_svd = trunk_model # Trained on SVD
        if trunk_modes > state_dim:
            self.trunk_extra = TrunkNet(in_size=256,out_size=self.trunk_modes-self.state_dim,hidden_size=self.trunk_size_extra,use_batch_norm=False,dropout_prob=0.1)
        else: 
            self.trunk_extra = None

        ### Nonlinear decoder (MLP) ###
        self.output_NN = FFNet(in_size=trunk_modes,out_size = 1,hidden_size=self.NL_size,use_batch_norm=use_batch_norm)

    def forward(self, x, rnn_input,locations, deltas):

        ### Projection ###
        if self.regular_enabled == False:
            unpadded_u, unpacked_lengths = pad_packed_sequence(rnn_input, batch_first=True)     # unpack input
            u = unpadded_u[:, :, :-1]                                                           # extract inputs values
            if self.trunk_extra is not None:                 
                trunk_output_svd = self.trunk_svd(locations.view(-1, 1)) 
                trunk_output_extra = self.trunk_extra(locations.view(-1, 1))
                trunk_output = torch.cat([trunk_output_svd, trunk_output_extra], dim=1)  
            else:
                trunk_output = self.trunk_svd(locations.view(-1, 1)) 
            x = torch.einsum("ni,bn->bi",trunk_output,x) # a(0)
            u = torch.einsum('ni,btn->bti',trunk_output,u) # projected inputs
            u_deltas = torch.cat((u, deltas), dim=-1)          
            rnn_input = pack_padded_sequence(u_deltas, unpacked_lengths, batch_first=True)      # repack RNN input

        
        ### Flow encoder ###
        h0 = self.x_dnn(x)
        if self.IC_encoder_decoder_enabled:
            x_repeated = x.repeat(1,self.control_rnn_depth)
            x_chunks = torch.chunk(x_repeated,chunks=self.control_rnn_depth, dim=1)
            enc_chunks = torch.chunk(h0, chunks=self.control_rnn_depth, dim=1)
            h0 = [torch.cat([b_part, a_part], dim=1) for b_part, a_part in zip(x_chunks, enc_chunks)]
            h0 = torch.stack(h0)
        else:
            h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.zeros_like(h0)

        ### Flow RNN ###
        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, h0)
        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed,
                                                           batch_first=True)
        h_shift = torch.roll(h, shifts=1, dims=1)
        h_shift[:, 0, :] = h0[-1]
        encoded_controls = (1 - deltas) * h_shift + deltas * h

        ### Flow decoder ###
        if self.IC_encoder_decoder_enabled:
            output_flow = encoded_controls[range(encoded_controls.shape[0]),
                                             h_lens - 1, :][:,:self.trunk_modes]
                     
        else:
            output_flow = self.u_dnn(encoded_controls[range(encoded_controls.shape[0]),
                                             h_lens - 1, :])
        
        ### Nonlinearity ###
        if self.nonlinear_enabled:
            output = torch.einsum("ni,bi->bni",trunk_output,output_flow)
            batch_size = output.shape[0]
            output = self.output_NN(output)
            output = output.view(batch_size,-1)

        ### Inner product ###
        else: 
            output = torch.einsum("ni,bi->bn",trunk_output,output_flow)
        return output, trunk_output

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
            self.layers.append(nn.Dropout(0.1))  # Dropout layer

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
                 dropout_prob=0.0,
                 activation=nn.Tanh):
        super(TrunkNet, self).__init__()

       
        B = torch.normal(mean=0.0, std=100, size=(128, 1))
        self.register_buffer("B", B)  

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size[0]))
        self.layers.append(activation())
        
        if use_batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_size[0]))

        if dropout_prob > 0:
            self.layers.append(nn.Dropout(dropout_prob))

        for isz, osz in zip(hidden_size[:-1], hidden_size[1:]):
            self.layers.append(nn.Linear(isz, osz))
            self.layers.append(activation())
            # self.layers.append(nn.Dropout(0.01))  # Dropout layer

            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(osz))
            if dropout_prob > 0:
                self.layers.append(nn.Dropout(dropout_prob))


        self.layers.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, input):
        input_proj = 2 * torch.pi * input @ self.B.T
        input = torch.cat([torch.sin(input_proj), torch.cos(input_proj)], dim=-1)
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

class CNN_encoder(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 use_batch_norm,
                 activation=nn.ReLU):
        super(CNN_encoder, self).__init__()

        self.dropout = nn.Dropout(p=0.3)  
        self.input_dim = in_size
        self.output_dim = out_size

        self.layers = nn.ModuleList()
        conv_channels = [1, 16, 32]

        # Convolutional layers with gradual max pooling
        for i, (isz, osz) in enumerate(zip(conv_channels[:-1], conv_channels[1:])):
            self.layers.append(nn.Conv1d(in_channels=isz, 
                                         out_channels=osz, 
                                         kernel_size=3,  
                                         stride=1, 
                                         padding=1))  
            
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(osz))
            
            self.layers.append(activation())

            # Apply MaxPooling every 3 layers
            if i % 3 == 0:
                self.layers.append(nn.MaxPool1d(kernel_size=2, stride=2))  # Reduce size gradually

        self.fc = nn.Linear(conv_channels[-1] * (in_size//2), self.output_dim)  # Adjust output size
     
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        # Apply convolutional layers with pooling
        for layer in self.layers:
            x = layer(x)

        # Flatten and pass through fully connected layer
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

