import torch.nn as nn
import torch
import logging

from model.utils.attention_blocks import SEBlock,SelfAttentionBlock
from model.utils.chunks import CNN1DTBlock,CNN2DBlock,CNN1DTSBlock,Classifier,ShortCut1DT,ShortCut2D,ShortCut1DTS


class AttentionCNN(nn.Module):
    def __init__(self,input_shape,output_shape,depth,mode,path,use_residual=True,
                 nb_features = 64 ,kernel_size=(64,64), saveModel_suffix='',
                 multitask=False,
                 use_SEB = False, use_self_attention=False):
        super().__init__()


        self.input_shape = input_shape # (n_electrodes,n_timepoints)
        self.nb_electrodes = self.input_shape[0]
        self.nb_timepoints = self.input_shape[1]
        self.output_shape = output_shape

        self.depth = depth # number of ConvBlock
        if depth % 3 != 0:
            logging.info('!!! WARNING: depth is not a multiple of 3 !!!')
        self.nb_features = nb_features
        self.kernel_size = kernel_size
        self.use_residual = use_residual
        self.use_SEB = use_SEB
        self.use_self_attention = use_self_attention


        self.mode = mode # mode = '1DT', '2D' or '1DST'
        self.path = path
        self.saveModel_suffix = saveModel_suffix
        self.multitask = multitask

        self.new_model =False

        if self.new_model:
            self.nb_features = 8
            self.kernel_size = (3,3)




        logging.info(f"-----PARAMETERS FOR CNNMULTITASK-----")
        logging.info(f"\tNEW MODEL              : {self.new_model}")
        logging.info(f"\tMode                   : {self.mode}")
        logging.info(f"\tSave Model Suffix      : {self.saveModel_suffix}")
        logging.info(f"\tInput Shape            : {self.input_shape}")
        logging.info(f"\tOutput Shape           : {self.output_shape}")

        logging.info(f"\tDepth                  : {self.depth}")
        logging.info(f"\tKernel Size            : {self.kernel_size}")
        logging.info(f"\tNumber of Features (Middle layer channel)      : {self.nb_features}")
        logging.info(f"\tUse Residual           : {self.use_residual}")
        logging.info(f"\tMultitask              : {self.multitask}")
        logging.info(f"\tUse SE Block           : {self.use_SEB}")
        logging.info(f"\tUse Self Attention     : {self.use_self_attention}")


        if self.mode == '1DT':
            if self.new_model:
                self.conv_blocks = nn.ModuleList(
                    [CNN1DTBlock(in_channels=1 if i == 0 else self.nb_features,
                                 out_channels=self.nb_features,
                                 kernel_size=self.kernel_size[0]) for i in range(self.depth)])
                self.shortcuts = nn.ModuleList(
                    [ShortCut1DT(in_channels=1 if i == 0 else self.nb_features,
                                 out_channels=self.nb_features) for i in
                     range(int(self.depth / 3))])
                self.output_layer = nn.Linear(in_features=self.nb_features * self.nb_electrodes * self.nb_timepoints,
                                              out_features=self.output_shape)
                if self.use_SEB:
                    self.se_layer = SEBlock(nb_channels=self.nb_electrodes,nb_features=self.nb_features * self.nb_timepoints)
                if self.use_self_attention:
                    self.self_attention_layer = SelfAttentionBlock(nb_features=self.nb_features * self.nb_timepoints)


            else:
                self.conv_blocks = nn.ModuleList([CNN1DTBlock(in_channels=self.nb_timepoints if i == 0 else self.nb_features,
                                                             out_channels=self.nb_features,
                                                             kernel_size=self.kernel_size[0]) for i in range(self.depth)])
                self.shortcuts = nn.ModuleList([ShortCut1DT(in_channels=self.nb_timepoints if i == 0 else self.nb_features,
                                                         out_channels=self.nb_features) for i in
                                                range(int(self.depth / 3))])
                self.output_layer = nn.Linear(in_features=self.nb_features * self.nb_electrodes,
                                              out_features=self.output_shape)
                self.classifier = Classifier(input_features=self.nb_features * self.nb_electrodes,
                                            hidden_para=128,num_class=72)
                if self.use_SEB:
                    self.se_layer = SEBlock(nb_channels=self.nb_electrodes,nb_features=self.nb_features)
                if self.use_self_attention:
                    self.self_attention_layer = SelfAttentionBlock(nb_features=self.nb_features)

                self.gap_layer = nn.AvgPool1d(kernel_size=2, stride=1)

        elif self.mode == '1DTS':
            self.conv_blocks = nn.ModuleList([CNN1DTSBlock(in_channels=(self.nb_timepoints,self.nb_electrodes) if i ==0 else (self.nb_features,self.nb_features),
                                                           out_channels=self.nb_features,
                                                           kernel_size=(32,16)) for i in range(self.depth)])
            self.shortcuts = nn.ModuleList([ShortCut1DTS(in_channels=(self.nb_timepoints,self.nb_electrodes) if i ==0 else (self.nb_features,self.nb_features),
                                                         out_channels=self.nb_features) for i in range(int(self.depth / 3))])
            self.output_layer = nn.Linear(in_features=self.nb_features * self.nb_features,
                                          out_features=self.output_shape)
            self.classifier = Classifier(input_features=self.nb_features * self.nb_features,
                                         hidden_para=128, num_class=72)
            if self.use_SEB:
                self.se_layer = SEBlock(nb_channels=self.nb_features,nb_features=self.nb_features)
            if self.use_self_attention:
                self.self_attention_layer = SelfAttentionBlock(nb_features=self.nb_features)


        elif self.mode == '2D':
            self.conv_blocks = nn.ModuleList([CNN2DBlock(
                in_channels=1 if i == 0 else self.nb_features,
                out_channels=self.nb_features,
                kernel_size=(32, 16)) for i in range(self.depth)])
            self.shortcuts = nn.ModuleList([ShortCut2D(
                in_channels=1 if i == 0 else self.nb_features,
                out_channels=self.nb_features) for i in range(int(self.depth / 3))])
            self.output_layer = nn.Linear(in_features=self.nb_features * self.nb_timepoints * self.nb_electrodes,
                                          out_features=self.output_shape)
        else:
            raise Exception('Not implemented.')



    def forward(self,x):
        # x : (batch_size, timepoints, num_electrodes)

        current_batch_size = x.shape[0]
        if self.mode == '2D':
            x = torch.unsqueeze(x,dim=1)

        if self.new_model:
            x = torch.reshape(x, (current_batch_size * self.nb_electrodes, 1, self.nb_timepoints))
        # x = torch.permute(x,(0,2,1)) # x : (batch_size,num_electrodes,timepoints)
        input_res = x  # set for the residual shortcut connection
        # Stack the modules and residual connection
        shortcut_cnt = 0


        for d in range(self.depth):
            x = self.conv_blocks[d](x)
            if self.use_residual and d % 3 == 2:
                res = self.shortcuts[shortcut_cnt](input_res)
                shortcut_cnt += 1
                if self.use_SEB:
                    # TODO
                    if self.new_model:
                        x = torch.reshape(x,(current_batch_size,self.nb_electrodes,self.nb_timepoints*self.nb_features))
                        x = torch.transpose(x,2,1)
                    x, _  = self.se_layer(x)
                    if self.new_model:
                        x = torch.transpose(x,2,1)
                        x = torch.reshape(x,(current_batch_size*self.nb_electrodes,self.nb_features,self.nb_timepoints))
                x = torch.add(x, res)
                x = nn.functional.relu(x)
                input_res = x
        # x: (batch_size, num_electrodes, nb_features)
        # x = self.gap_layer(x)
        if self.use_self_attention:
            if self.new_model:
                x = torch.reshape(x, (current_batch_size, self.nb_electrodes, self.nb_timepoints * self.nb_features))
                x = torch.transpose(x,2,1)
            x, _ = self.self_attention_layer(x)
            if self.new_model:
                x = torch.transpose(x,2,1)
                x = torch.reshape(x, (current_batch_size * self.nb_electrodes, self.nb_features, self.nb_timepoints))

        x = x.reshape(current_batch_size, -1)
        output = self.output_layer(x)  # Defined in BaseNet
        if self.multitask:
            id = self.classifier(x)
            return output, id
        return output

    def save(self):
        ckpt_dir = self.path + 'CNNMultiTask' + \
            '_nb_{}_{}'.format(0,self.saveModel_suffix) + '.pth'
        torch.save(self.state_dict(), ckpt_dir)
        logging.info(f"Saved new best model (on validation data) to ckpt_dir")

    def predict(self, x):
        # x : (batch_size, timepoints, num_electrodes)
        scale = None
        current_batch_size = x.shape[0]
        if self.mode == '2D':
            x = torch.unsqueeze(x, dim=1)

        if self.new_model:
            x = torch.reshape(x, (current_batch_size * self.nb_electrodes, 1, self.nb_timepoints))
        # x = torch.permute(x,(0,2,1)) # x : (batch_size,num_electrodes,timepoints)
        input_res = x  # set for the residual shortcut connection
        # Stack the modules and residual connection
        shortcut_cnt = 0

        for d in range(self.depth):
            x = self.conv_blocks[d](x)
            if self.use_residual and d % 3 == 2:
                res = self.shortcuts[shortcut_cnt](input_res)
                shortcut_cnt += 1
                if self.use_SEB:
                    # TODO
                    if self.new_model:
                        x = torch.reshape(x, (
                        current_batch_size, self.nb_electrodes, self.nb_timepoints * self.nb_features))
                        x = torch.transpose(x, 2, 1)
                    x, scale = self.se_layer(x)
                    if self.new_model:
                        x = torch.transpose(x, 2, 1)
                        x = torch.reshape(x, (
                        current_batch_size * self.nb_electrodes, self.nb_features, self.nb_timepoints))
                x = torch.add(x, res)
                x = nn.functional.relu(x)
                input_res = x
        # x: (batch_size, num_electrodes, nb_features)
        # x = self.gap_layer(x)
        if self.use_self_attention:
            if self.new_model:
                x = torch.reshape(x, (current_batch_size, self.nb_electrodes, self.nb_timepoints * self.nb_features))
                x = torch.transpose(x, 2, 1)
            x, scale = self.self_attention_layer(x)
            if self.new_model:
                x = torch.transpose(x, 2, 1)
                x = torch.reshape(x, (current_batch_size * self.nb_electrodes, self.nb_features, self.nb_timepoints))

        x = x.reshape(current_batch_size, -1)
        output = self.output_layer(x)  # Defined in BaseNet
        if self.multitask:
            id = self.classifier(x)
            return output, id
        return output,scale


if __name__ == '__main__':
    model = AttentionCNN(input_shape=(129,500),output_shape=2,depth=12,mode='1DT',path= '',use_residual=True,
                 nb_features = 64 ,kernel_size=(64,64), saveModel_suffix='',multitask=False,
                 use_SEB = False, use_self_attention=False)
    print(model)