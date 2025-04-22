import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from utils import dummy_context_mgr


class CLIP_IMG_ENCODER(nn.Module):
    """
       CLIP_IMG_ENCODER module for encoding images using CLIP's visual transformer.
    """

    def __init__(self, CLIP):
        """
        Initialize the CLIP_IMG_ENCODER module.

        Args:
            CLIP (CLIP): Pre-trained CLIP model.
        """
        super(CLIP_IMG_ENCODER, self).__init__()
        model = CLIP.visual
        self.define_module(model)
        # freeze the parameters of the CLIP model
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, model):
        """
        Define the individual layers and modules of the CLIP visual transformer model.
        Args:
            model (nn.Module): CLIP visual transformer model.
        """
        # Extract required modules from the CLIP model
        self.conv1 = model.conv1  # Convolutional layer
        self.class_embedding = model.class_embedding  # Class embedding layer
        self.positional_embedding = model.positional_embedding  # Positional embedding layer
        self.ln_pre = model.ln_pre  # Linear Normalization layer for pre-normalization
        self.transformer = model.transformer  # Transformer block
        self.ln_post = model.ln_post  # Linear Normalization layer for post-normalization
        self.proj = model.proj  # projection matrix

    @property
    def dtype(self):
        """
         Get the data type of the convolutional layer weights.
        """
        return self.conv1.weight.dtype

    def transf_to_CLIP_input(self, inputs):
        """
        Transform input images to the format expected by CLIP.

        Args:
            inputs (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Transformed images.
        """
        device = inputs.device
        # Check the size of the input image tensor
        if len(inputs.size()) != 4:
            raise ValueError('Expect the (B, C, X, Y) tensor.')
        else:
            # Normalize input images
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            var = torch.tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            inputs = F.interpolate(inputs * 0.5 + 0.5, size=(224, 224))
            inputs = ((inputs + 1) * 0.5 - mean) / var
            return inputs

    def forward(self, img: torch.Tensor):
        """
        Forward pass of the CLIP_IMG_ENCODER module.

        Args:
            img (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Local features extracted from the image.
            torch.Tensor: Encoded image embeddings.
        """
        # Transform input images to the format expected by CLIP and set its datatype appropriately
        x = self.transf_to_CLIP_input(img)
        x = x.type(self.dtype)

        # Pass the image through Convolutional layer
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid = x.size(-1)

        # Reshape and permute the tensor for transformer input
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # Add class and positional embeddings
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # NLD (Batch Size - Length - Dimension) -> LND (Length - Batch Size - Dimension)
        x = x.permute(1, 0, 2)

        # Extract local features using transformer blocks
        selected = [1, 4, 8]
        local_features = []
        for i in range(12):
            x = self.transformer.resblocks[i](x)
            if i in selected:
                local_features.append(
                    x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(
                        img.dtype))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj  # Perform matrix multiplication with projection matrix and tensor
        return torch.stack(local_features, dim=1), x.type(img.dtype)


class CLIP_TXT_ENCODER(nn.Module):
    """
        CLIP_TXT_ENCODER module for encoding text inputs using CLIP's transformer.
    """

    def __init__(self, CLIP):
        """
        Initialize the CLIP_TXT_ENCODER module.

        Args:
            CLIP (CLIP): Pre-trained CLIP model.
        """
        super(CLIP_TXT_ENCODER, self).__init__()
        self.define_module(CLIP)
        # Freeze the parameters of the CLIP model
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, CLIP):
        """
        Define the individual modules of the CLIP transformer model.

        Args:
            CLIP (CLIP): Pre-trained CLIP model.
        """
        self.transformer = CLIP.transformer  # Transformer block
        self.vocab_size = CLIP.vocab_size  # Size of the vocabulary of the transformer
        self.token_embedding = CLIP.token_embedding  # token embedding block
        self.positional_embedding = CLIP.positional_embedding  # positional embedding block
        self.ln_final = CLIP.ln_final  # Linear Normalization layer
        self.text_projection = CLIP.text_projection  # Projection matrix for text

    @property
    def dtype(self):
        """
        Get the data type of the first layer's weights in the transformer.
        """
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, text):
        """
        Forward pass of the CLIP_TXT_ENCODER module.

        Args:
            text (torch.Tensor): Input text tokens.

        Returns:
            torch.Tensor: Encoded sentence embeddings.
            torch.Tensor: Transformer output for the input text.
        """
        # Embed input text tokens
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # Add positional embeddings
        x = x + self.positional_embedding.type(self.dtype)
        # Permute dimensions for transformer input
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass input through the transformer
        x = self.transformer(x)
        # Permute dimensions back to original shape
        x = x.permute(1, 0, 2)  # LND -> NLD
        # Apply layer normalization
        x = self.ln_final(x).type(self.dtype)  # shape = [batch_size, n_ctx, transformer.width]
        # Extract sentence embeddings from the end-of-text (eot_token : is the highest number in each sequence)
        sent_emb = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        # Return the sentence embedding and transformer ouput
        return sent_emb, x


class CLIP_Mapper(nn.Module):
    """
    CLIP_Mapper module for mapping images with prompts using CLIP's transformer.
    """

    def __init__(self, CLIP):
        """
        Initialize the CLIP_Mapper module.

        Args:
            CLIP (CLIP): Pre-trained CLIP model.
        """
        super(CLIP_Mapper, self).__init__()
        model = CLIP.visual
        self.define_module(model)
        # Freeze the parameters of the CLIP visual model
        for param in model.parameters():
            param.requires_grad = False

    def define_module(self, model):
        """
        Define the individual modules of the CLIP visual model.

        Args:
            model: Pre-trained CLIP visual model.
        """
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = model.transformer

    @property
    def dtype(self):
        """
        Get the data type of the weights of the first convolutional layer.
        """
        return self.conv1.weight.dtype

    def forward(self, img: torch.Tensor, prompts: torch.Tensor):
        """
        Forward pass of the CLIP_Mapper module.

        Args:
            img (torch.Tensor): Input image tensor.
            prompts (torch.Tensor): Prompt tokens for mapping.

        Returns:
            torch.Tensor: Mapped features from the CLIP model.
        """

        # Convert input image and prompts to the appropriate data type
        x = img.type(self.dtype)
        prompts = prompts.type(self.dtype)
        grid = x.size(-1)

        # Reshape the input image tensor
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # Append the class embeddings to input tensors
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x],
            dim=1
        )  # shape = [*, grid ** 2 + 1, width]

        # Append the positional embeddings to the input tensor
        x = x + self.positional_embedding.to(x.dtype)

        # Perform the layer normalization
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        # Local features
        selected = [1, 2, 3, 4, 5, 6, 7, 8]
        begin, end = 0, 12
        prompt_idx = 0
        for i in range(begin, end):
            # Add prompt to the input tensor
            if i in selected:
                prompt = prompts[:, prompt_idx, :].unsqueeze(0)
                prompt_idx = prompt_idx + 1
                x = torch.cat((x, prompt), dim=0)
                x = self.transformer.resblocks[i](x)
                x = x[:-1, :, :]
            else:
                x = self.transformer.resblocks[i](x)
        # Reshape and return mapped features
        return x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(img.dtype)


class CLIP_Adapter(nn.Module):
    """
    CLIP_Adapter module for adapting features from a generator to match the CLIP model's input requirements.
    """

    def __init__(self, in_ch, mid_ch, out_ch, G_ch, CLIP_ch, cond_dim, k, s, p, map_num, CLIP):
        """
        Initialize the CLIP_Adapter module.

        Args:
            in_ch (int): Number of input channels.
            mid_ch (int): Number of channels in the intermediate layers.
            out_ch (int): Number of output channels.
            G_ch (int): Number of channels in the generator's output.
            CLIP_ch (int): Number of channels in the CLIP model's input.
            cond_dim (int): Dimension of the conditioning vector.
            k (int): Kernel size for convolutional layers.
            s (int): Stride for convolutional layers.
            p (int): Padding for convolutional layers.
            map_num (int): Number of mapping blocks.
            CLIP: Pre-trained CLIP model.
        """
        super(CLIP_Adapter, self).__init__()
        self.CLIP_ch = CLIP_ch
        self.FBlocks = nn.ModuleList([])
        # Define Mapping blocks (M_Block) and them to Feature blocks (FBlock) for given number of mapping blocks.
        self.FBlocks.append(M_Block(in_ch, mid_ch, out_ch, cond_dim, k, s, p))
        for i in range(map_num - 1):
            self.FBlocks.append(M_Block(out_ch, mid_ch, out_ch, cond_dim, k, s, p))
        # Convolutional layer to fuse adapted features
        self.conv_fuse = nn.Conv2d(out_ch, CLIP_ch, 5, 1, 2)
        # CLIP Mapper module to map adapted features to CLIP's input space
        self.CLIP_ViT = CLIP_Mapper(CLIP)
        # Convolutional layer to further process mapped features
        self.conv = nn.Conv2d(768, G_ch, 5, 1, 2)
        # Fully connected layer for conditioning
        self.fc_prompt = nn.Linear(cond_dim, CLIP_ch * 8)

    def forward(self, out, c):
        """
        Forward pass of the CLIP_Adapter module. Takes output features from the generator and conditioning vector
        as input, adapts features using the Feature block having multiple mapping blocks, fuses them, map them to
        CLIPs input space and returns the processed features

        Args:
            out (torch.Tensor): Output features from the generator.
            c (torch.Tensor): Conditioning vector.

        Returns:
            torch.Tensor: Adapted and mapped features for the generator.
        """

        # Generate prompts from the conditioning vector
        prompts = self.fc_prompt(c).view(c.size(0), -1, self.CLIP_ch)

        # Pass features through feature block consisting of multiple mapping blocks
        for FBlock in self.FBlocks:
            out = FBlock(out, c)
        # Fuse adapted features
        fuse_feat = self.conv_fuse(out)
        # Map fused features to CLIP's input space
        map_feat = self.CLIP_ViT(fuse_feat, prompts)
        # Further process mapped features and return
        return self.conv(fuse_feat + 0.1 * map_feat)


class NetG(nn.Module):
    """
    Generator network for synthesizing images conditioned on text and noise
    """

    def __init__(self, ngf, nz, cond_dim, imsize, ch_size, mixed_precision, CLIP):
        """
        Initializes the Generator network.

        Parameters:
            ngf (int): Number of generator filters.
            nz (int): Dimensionality of the input noise vector.
            cond_dim (int): Dimensionality of the conditioning vector.
            imsize (int): Size of the generated images.
            ch_size (int): Number of output channels for the generated images.
            mixed_precision (bool): Whether to use mixed precision training.
            CLIP: CLIP model for feature adaptation.

        """
        super(NetG, self).__init__()
        # Define attributes
        self.ngf = ngf
        self.mixed_precision = mixed_precision

        # Build CLIP Mapper
        self.code_sz, self.code_ch, self.mid_ch = 7, 64, 32
        self.CLIP_ch = 768
        # fully connected layer to convert the noise vector into a feature map of dimensions (code_sz * code_sz * code_ch)
        self.fc_code = nn.Linear(nz, self.code_sz * self.code_sz * self.code_ch)
        self.mapping = CLIP_Adapter(self.code_ch, self.mid_ch, self.code_ch, ngf * 8, self.CLIP_ch, cond_dim + nz, 3, 1,
                                    1, 4, CLIP)
        # Build GBlocks
        self.GBlocks = nn.ModuleList([])
        in_out_pairs = list(get_G_in_out_chs(ngf, imsize))
        imsize = 4
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            if idx < (len(in_out_pairs) - 1):
                imsize = imsize * 2
            else:
                imsize = 224
            self.GBlocks.append(G_Block(cond_dim + nz, in_ch, out_ch, imsize))

        # To RGB image conversion using the sequential layers having leakyReLU activation function
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, ch_size, 3, 1, 1),
        )

    def forward(self, noise, c, eval=False):  # x=noise, c=ent_emb
        """
        Forward pass of the generator network.

        Args:
            noise (torch.Tensor): Input noise vector.
            c (torch.Tensor): Conditioning information, typically an embedding representing attributes of the output.
            eval (bool, optional): Flag indicating whether the network is in evaluation mode. Defaults to False.

        Returns:
            torch.Tensor: Generated RGB images.
        """
        # Context manager for enabling automatic mixed precision training
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else dummy_context_mgr() as mp:
            # Concatenate noise and conditioning information
            cond = torch.cat((noise, c), dim=1)

            # Pass noise through fully connected layer to generate feature map and adapt features using CLIP Mapper
            out = self.mapping(self.fc_code(noise).view(noise.size(0), self.code_ch, self.code_sz, self.code_sz), cond)

            # Apply GBlocks to progressively upsample feature representation, fuse text and visual features
            for GBlock in self.GBlocks:
                out = GBlock(out, cond)

            # Convert final feature representation to RGB images
            out = self.to_rgb(out)

        return out


class NetD(nn.Module):
    """
    Discriminator network for evaluating the realism of images.
    Attributes:
        DBlocks (nn.ModuleList): List of D_Block modules for processing feature maps.
        main (D_Block): Main D_Block module for final processing.
    """

    def __init__(self, ndf, imsize, ch_size, mixed_precision):
        """
        Initializes the Discriminator network

        Args:
        ndf (int): Number of channels in the initial features.
        imsize (int): Size of the input images (assumed square).
        ch_size (int): Number of channels in the output feature maps.
        mixed_precision (bool): Flag indicating whether to use mixed precision training.
        """
        super(NetD, self).__init__()
        self.mixed_precision = mixed_precision
        # Define the DBlock
        self.DBlocks = nn.ModuleList([
            D_Block(768, 768, 3, 1, 1, res=True, CLIP_feat=True),
            D_Block(768, 768, 3, 1, 1, res=True, CLIP_feat=True),
        ])
        # Define the main DBlock for the final processing
        self.main = D_Block(768, 512, 3, 1, 1, res=True, CLIP_feat=False)

    def forward(self, h):
        """
        Forward pass of the discriminator network.
        Args:
            h (torch.Tensor): Input feature maps.
        Returns:
            torch.Tensor: Discriminator output.
        """
        with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
            # Initial feature map
            out = h[:, 0]
            # Pass the input feature through each DBlock
            for idx in range(len(self.DBlocks)):
                out = self.DBlocks[idx](out, h[:, idx + 1])
            # Final processing through the main DBlock
            out = self.main(out)
        return out


class NetC(nn.Module):
    """
    Classifier / Comparator network for classifying the joint features of the generator output and condition text.
    Attributes:
        cond_dim (int): Dimensionality of the conditioning information.
        mixed_precision (bool): Flag indicating whether to use mixed precision training.
        joint_conv (nn.Sequential): Sequential module defining the classifier layers.
    """
    def __init__(self, ndf, cond_dim, mixed_precision):
        """

        """
        super(NetC, self).__init__()
        self.cond_dim = cond_dim
        self.mixed_precision = mixed_precision
        # Define the classifier layers, sequential convolutional 2D layer with LeakyReLU as the activation function
        self.joint_conv = nn.Sequential(
            nn.Conv2d(512 + 512, 128, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, cond):
        """
        Forward pass of the classifier network.

        Args:
            out (torch.Tensor): Generator output feature map.
            cond (torch.Tensor): Conditioning information vector
        """
        with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
            # Reshape and repeat conditioning information vector to match the feature map size
            cond = cond.view(-1, self.cond_dim, 1, 1)
            cond = cond.repeat(1, 1, 7, 7)

            # Concatenate feature map and conditioned information
            h_c_code = torch.cat((out, cond), 1)

            # Pass through the classifier layers
            out = self.joint_conv(h_c_code)
        return out


class M_Block(nn.Module):
    """
    Multi-scale block consisting of convolutional layers and conditioning.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        fuse1 (DFBlock): Conditioning block for the first convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        fuse2 (DFBlock): Conditioning block for the second convolutional layer.
        learnable_sc (bool): Flag indicating whether the shortcut connection is learnable.
        c_sc (nn.Conv2d): Convolutional layer for the shortcut connection.

    """
    def __init__(self, in_ch, mid_ch, out_ch, cond_dim, k, s, p):
        """
        Initializes the Multi-scale block.

        Args:
            in_ch (int): Number of input channels.
            mid_ch (int): Number of channels in the intermediate layers.
            out_ch (int): Number of output channels.
            cond_dim (int): Dimensionality of the conditioning information.
            k (int): Kernel size for convolutional layers.
            s (int): Stride for convolutional layers.
            p (int): Padding for convolutional layers.

        """
        super(M_Block, self).__init__()

        # Define convolutional layers and conditioning blocks
        self.conv1 = nn.Conv2d(in_ch, mid_ch, k, s, p)
        self.fuse1 = DFBLK(cond_dim, mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, k, s, p)
        self.fuse2 = DFBLK(cond_dim, out_ch)

        # Learnable shortcut connection
        self.learnable_sc = in_ch != out_ch
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        """
        Defines the shortcut connection.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Shortcut connection output.
        """
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, text):
        """
        Defines the residual path with conditioning.

        Args:
            h (torch.Tensor): Input tensor.
            text (torch.Tensor): Conditioning information.

        Returns:
            torch.Tensor: Residual path output.
        """
        h = self.conv1(h)
        h = self.fuse1(h, text)
        h = self.conv2(h)
        h = self.fuse2(h, text)
        return h

    def forward(self, h, c):
        """
        Forward pass of the multi-scale block.

        Args:
            h (torch.Tensor): Input tensor.
            c (torch.Tensor): Conditioning information.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.shortcut(h) + self.residual(h, c)


class G_Block(nn.Module):
    """
        Generator block consisting of convolutional layers and conditioning.

        Attributes:
            imsize (int): Size of the output image.
            learnable_sc (bool): Flag indicating whether the shortcut connection is learnable.
            c1 (nn.Conv2d): First convolutional layer.
            c2 (nn.Conv2d): Second convolutional layer.
            fuse1 (DFBLK): Conditioning block for the first convolutional layer.
            fuse2 (DFBLK): Conditioning block for the second convolutional layer.
            c_sc (nn.Conv2d): Convolutional layer for the shortcut connection.
        """

    def __init__(self, cond_dim, in_ch, out_ch, imsize):
        """
        Initialize the Generator block.

        Args:
            cond_dim (int): Dimensionality of the conditioning information.
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            imsize (int): Size of the output image.
        """
        super(G_Block, self).__init__()

        # Initialize attributes
        self.imsize = imsize
        self.learnable_sc = in_ch != out_ch

        # Define convolution layers and conditioning blocks
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)

        # Learnable shortcut connection
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        """
        Defines the shortcut connection.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Shortcut connection output.
        """
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y):
        """
        Defines the residual path with conditioning.

        Args:
            h (torch.Tensor): Input tensor.
            y (torch.Tensor): Conditioning information.

        Returns:
            torch.Tensor: Residual path output.
        """
        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)
        return h

    def forward(self, h, y):
        """
        Forward pass of the generator block.

        Args:
            h (torch.Tensor): Input tensor.
            y (torch.Tensor): Conditioning information.

        Returns:
            torch.Tensor: Output tensor.
        """
        h = F.interpolate(h, size=(self.imsize, self.imsize))
        return self.shortcut(h) + self.residual(h, y)


class D_Block(nn.Module):
    """
    Discriminator block.
    """
    def __init__(self, fin, fout, k, s, p, res, CLIP_feat):
        """
        Initializes Discriminator block.

        Args:
        - fin (int): Number of input channels.
        - fout (int): Number of output channels.
        - k (int): Kernel size for convolutional layers.
        - s (int): Stride for convolutional layers.
        - p (int): Padding for convolutional layers.
        - res (bool): Whether to use residual connection.
        - CLIP_feat (bool): Whether to incorporate CLIP features.
        """
        super(D_Block, self).__init__()
        self.res, self.CLIP_feat = res, CLIP_feat
        self.learned_shortcut = (fin != fout)

        # Convolutional layers for residual path
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Convolutional layers for shortcut connection
        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)

        # Parameters for learned residual and CLIP features
        if self.res == True:
            self.gamma = nn.Parameter(torch.zeros(1))
        if self.CLIP_feat == True:
            self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, CLIP_feat=None):
        """
        Forward pass of the discriminator block.

        Args:
        - x (torch.Tensor): Input tensor.
        - CLIP_feat (torch.Tensor): Optional CLIP features tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        # Compute the residual features
        res = self.conv_r(x)

        # Compute the shortcut connection
        if self.learned_shortcut:
            x = self.conv_s(x)

        # Incorporate learned residual and CLIP features if enabled
        if (self.res == True) and (self.CLIP_feat == True):
            return x + self.gamma * res + self.beta * CLIP_feat
        elif (self.res == True) and (self.CLIP_feat != True):
            return x + self.gamma * res
        elif (self.res != True) and (self.CLIP_feat == True):
            return x + self.beta * CLIP_feat
        else:
            return x


class DFBLK(nn.Module):
    """
    Diffusion Block of the Generator network with Conditional feature block
    """
    def __init__(self, cond_dim, in_ch):
        """
        Initializing the Conditional feature block of the DFBlock.

        Args:
        - cond_dim (int): Dimensionality of the conditional input.
        - in_ch (int): Number of input channels.
        """
        super(DFBLK, self).__init__()
        # Define conditional affine transformations
        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)

    def forward(self, x, y=None):
        """
        Forward pass of the conditional feature block.

        Args:
        - x (torch.Tensor): Input tensor.
        - y (torch.Tensor, optional): Conditional input tensor. Default is None.

        Returns:
        - torch.Tensor: Output tensor.
        """
        # Apply the first affine transformation and activation function
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        # Apply second affine transformation and activation function
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        return h


class QuickGELU(nn.Module):
    """
    Efficient and faster version of GELU,
    for non-linearity and to learn complex patterns
    """
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the QuickGELU activation function.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        # Apply QuickGELU activation function
        return x * torch.sigmoid(1.702 * x)


# Taken from the RAT-GAN repository
class Affine(nn.Module):
    """
    Affine transformation module that applies conditional scaling and shifting to input features,
    to incorporate additional control over the generated output based on input conditions.
    """
    def __init__(self, cond_dim, num_features):
        """
        Initialize the affine transformation module.
        Args:
            cond_dim (int): Dimensionality of the conditioning information.
            num_features (int): Number of input features.
        """
        super(Affine, self).__init__()
        # Define 2 fully connected networks to compute gamma and beta parameters
        # each 2 linear layers with RELU activation in between
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, num_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, num_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features, num_features)),
        ]))
        # Initializes the weights and biases of the network
        self._initialize()

    def _initialize(self):
        """
        Initializes the weights and biases of the linear layers responsible for computing gamma and beta
        """
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        """
        Forward pass of the Affine transformation module.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor, optional): Conditioning information tensor. Default is None.

        Returns:
            torch.Tensor: Transformed tensor after applying affine transformation.
        """
        # Compute gamma and beta parameters
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        # Ensure proper shape for weight and bias tensors
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        # Expand weight and bias tensors to match input tensor shape
        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        # Apply affine transformation
        return weight * x + bias


def get_G_in_out_chs(nf, imsize):
    """
    Compute input-output channel pairs for generator blocks based on given number of channels and image size.

    Args:
        nf (int): Number of input channels.
        imsize (int): Size of the input image.

    Returns:
        list: List of tuples containing input-output channel pairs for generator blocks.
    """
    # Determine the number of layers based on image size
    layer_num = int(np.log2(imsize)) - 1

    # Compute the number of channels for each layer
    channel_nums = [nf * min(2 ** idx, 8) for idx in range(layer_num)]

    # Reverse the channel numbers to start with the highest channel count
    channel_nums = channel_nums[::-1]

    # Generate input-output channel pairs for generator blocks
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])

    return in_out_pairs


def get_D_in_out_chs(nf, imsize):
    """
    Compute input-output channel pairs for discriminator blocks based on given number of channels and image size.

    Args:
        nf (int): Number of input channels.
        imsize (int): Size of the input image.

    Returns:
        list: List of tuples containing input-output channel pairs for discriminator blocks.
    """
    # Determine the number of layers based on image size
    layer_num = int(np.log2(imsize)) - 1

    # Compute the number of channels for each layer
    channel_nums = [nf * min(2 ** idx, 8) for idx in range(layer_num)]

    # Generate input-output channel pairs for discriminator blocks
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])

    return in_out_pairs
