# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, vq, vq_class, vq_dim, action_dim, use_vitg=False, vitg_ckpt_path=None, tactile_camera_names=None):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            use_vitg: whether to use ViTG encoder for tactile images
            vitg_ckpt_path: path to ViTG checkpoint file
            tactile_camera_names: list of tactile sensor names for ViTG (hybrid mode)
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names  # RGB cameras
        self.tactile_camera_names = tactile_camera_names if tactile_camera_names else []
        self.all_camera_names = camera_names + self.tactile_camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.vq, self.vq_class, self.vq_dim = vq, vq_class, vq_dim
        self.state_dim, self.action_dim = state_dim, action_dim
        self.use_vitg = use_vitg
        self.use_hybrid = use_vitg and len(self.tactile_camera_names) > 0 and backbones is not None
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        if self.use_hybrid:
            # Hybrid mode: ResNet backbones for RGB + ViTG for tactile
            print(f"Using HYBRID mode: ResNet for {len(camera_names)} RGB cameras + ViTG for {len(self.tactile_camera_names)} tactile sensors")
            
            # Create ResNet backbones for RGB cameras
            self.backbones = nn.ModuleList(backbones)
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            
            # Create ViTG encoder for tactile sensors (shared across all tactile sensors)
            from ModelTrain.module.vitg_encoder import ViTGEncoderSimple
            print(f"Loading ViTG from: {vitg_ckpt_path}")
            # Load once and share across all tactile sensors to save memory
            # ViT-Giant outputs 1408-dim embeddings
            self.vitg_encoder_shared = ViTGEncoderSimple(vitg_ckpt_path, embed_dim=1408, input_size=224)
            
            # Project ViTG embeddings (1408-dim) to hidden_dim
            self.vitg_proj = nn.Linear(1408, hidden_dim)
            
            # Position embedding for tactile features
            self.tactile_pos_embed = nn.Parameter(torch.randn(1, hidden_dim, 1))
            
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            
            
        elif backbones is not None:
            # Pure ResNet mode (RGB only)
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # No visual input
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding

        print(f'Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}')
        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        if self.vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent


    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        bs, _ = qpos.shape
        if self.encoder is None:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # cvae encoder
            is_training = actions is not None # train or val
            ### Obtain latent z from action sequence
            if is_training:
                # project action sequence to embedding dim, and concat with a CLS token
                action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                cls_embed = self.cls_embed.weight # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
                # do not mask cls token
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
                # obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
                # query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                encoder_output = encoder_output[0] # take cls output only
                latent_info = self.latent_proj(encoder_output)
                
                if self.vq:
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    probs = torch.softmax(logits, dim=-1)
                    binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                    latent_input = self.latent_out_proj(straigt_through)
                    mu = logvar = None
                else:
                    probs = binaries = None
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)

            else:
                mu = logvar = binaries = probs = None
                if self.vq:
                    latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                else:
                    latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    latent_input = self.latent_out_proj(latent_sample)

        return latent_input, probs, binaries, mu, logvar

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, vq_sample=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)

        # cvae decoder
        if self.use_hybrid:
            # Hybrid mode: Process RGB through ResNet + Tactile through ViTG
            bs = qpos.shape[0]
            all_features = []
            all_pos = []
            
            # Handle list input (for different resolutions) or tensor input
            if isinstance(image, list):
                rgb_images = image[0]  # (B, num_rgb, C, H, W)
                tactile_images = image[1]  # (B, num_tactile, C, H, W)
            else:
                # Single tensor input (same resolution for all)
                rgb_images = image[:, :len(self.camera_names)]
                tactile_images = image[:, len(self.camera_names):]
            
            # Process RGB cameras through ResNet
            for cam_id, cam_name in enumerate(self.camera_names):
                rgb_image = rgb_images[:, cam_id]
                features, pos = self.backbones[cam_id](rgb_image)
                features = features[0]  # take the last layer feature (B, C, H, W)
                pos = pos[0]
                
                #print(f"DEBUG cam {cam_id}: features.shape={features.shape}, pos.shape={pos.shape}")
                
                # Project and flatten to sequence
                projected = self.input_proj(features)  # (B, hidden_dim, H, W)
                projected = projected.flatten(2)  # (B, hidden_dim, H*W) - flatten spatial
                all_features.append(projected)
                #print(f"DEBUG cam {cam_id}: projected.shape after flatten={projected.shape}")
                
                # Flatten position embeddings
                pos = pos.flatten(2)  # (1, hidden_dim, H*W)
                all_pos.append(pos)
                #print(f"DEBUG cam {cam_id}: pos.shape after flatten={pos.shape}")
            
            # Process tactile sensors through ViTG (shared encoder)
            for tac_id, tac_name in enumerate(self.tactile_camera_names):
                tactile_image = tactile_images[:, tac_id]
                
                # Get ViTG embedding using shared encoder
                tac_embedding = self.vitg_encoder_shared(tactile_image)  # (B, 1408)
                #print(f"DEBUG tactile {tac_id}: tac_embedding.shape={tac_embedding.shape}")
                
                # Project to hidden_dim
                tac_feature = self.vitg_proj(tac_embedding)  # (B, hidden_dim)
                
                # Reshape to sequence format: (B, hidden_dim, 1) to match flattened RGB
                tac_feature = tac_feature.unsqueeze(-1)  # (B, hidden_dim, 1) - single token
                all_features.append(tac_feature)
                #print(f"DEBUG tactile {tac_id}: tac_feature.shape={tac_feature.shape}")
                
                # Position embedding as sequence: (1, hidden_dim, 1)
                tac_pos = self.tactile_pos_embed  # (1, hidden_dim, 1) - already correct shape
                all_pos.append(tac_pos)
                #print(f"DEBUG tactile {tac_id}: tac_pos.shape={tac_pos.shape}")
            
            # Concatenate all features along sequence dimension
            # RGB: (B, hidden_dim, 300) per camera × 3 = 900 tokens
            # Tactile: (B, hidden_dim, 1) per sensor × 1 = 1 token
            # Total: 901 tokens in unified sequence
            src = torch.cat(all_features, dim=2)  # (B, hidden_dim, total_seq_len=901)
            pos = torch.cat(all_pos, dim=2)  # (1, hidden_dim, total_seq_len=901)
            
            #print(f"DEBUG: src.shape after cat={src.shape}, pos.shape after cat={pos.shape}")
            
            # Reshape to 4D for transformer (it expects (B, C, H, W) format)
            # Treat sequence as width dimension: (B, hidden_dim, 1, seq_len)
            src = src.unsqueeze(2)  # (B, hidden_dim, 1, 901)
            pos = pos.unsqueeze(2)  # (1, hidden_dim, 1, 901)
            
            #print(f"DEBUG: Final 4D format - src={src.shape}, pos={pos.shape}")
            #print(f"DEBUG: Expected: src=(B, C, H, W)=(8, 512, 1, 901), pos=(1, 512, 1, 901)")
            
            # Proprioception and transformer
            proprio_input = self.input_proj_robot_state(qpos)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
            
        elif self.use_vitg:
            # Pure ViTG mode: Process tactile images through ViTG encoders
            bs = qpos.shape[0]
            all_tactile_features = []
            
            for cam_id, cam_name in enumerate(self.camera_names):
                # Get tactile image for this sensor
                tactile_image = image[:, cam_id]  # (B, C, H, W)
                
                # Pass through ViTG encoder to get global embedding
                tactile_embedding = self.vitg_encoders[cam_id](tactile_image)  # (B, 1280)
                
                # Project to hidden_dim
                tactile_feature = self.vitg_proj(tactile_embedding)  # (B, hidden_dim)
                
                # Reshape to add a spatial dimension: (B, hidden_dim, 1)
                tactile_feature = tactile_feature.unsqueeze(-1)  # (B, hidden_dim, 1)
                
                all_tactile_features.append(tactile_feature)
            
            # Concatenate tactile features along width dimension
            # This treats each tactile sensor as a "spatial location"
            src = torch.cat(all_tactile_features, dim=2)  # (B, hidden_dim, num_sensors)
            
            # Create position embeddings for tactile sensors
            pos = self.tactile_pos_embed.repeat(1, 1, len(self.camera_names))  # (1, hidden_dim, num_sensors)
            pos = pos.repeat(bs, 1, 1)  # (B, hidden_dim, num_sensors)
            
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            
            # Pass through transformer
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
            
        elif self.backbones is not None:
            # Pure ResNet mode: Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + state_dim
            # self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=self.action_dim, hidden_depth=2)  # src code
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=16, hidden_depth=2)  # change code
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 14 # TODO hardcode

    # Check if using ViTG
    use_vitg = getattr(args, 'use_vitg', False)
    vitg_ckpt_path = getattr(args, 'vitg_ckpt_path', None)
    tactile_camera_names = getattr(args, 'tactile_camera_names', [])

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    if use_vitg and len(tactile_camera_names) > 0:
        # Hybrid mode: Build ResNet backbones for RGB cameras only (not tactile)
        backbones = []
        for cam_name in args.camera_names:
            if cam_name not in tactile_camera_names:
                backbone = build_backbone(args)
                backbones.append(backbone)
        
        if len(backbones) == 0:
            backbones = None  # Pure ViTG mode (no RGB cameras)
        
        print(f"Building HYBRID model: {len(backbones) if backbones else 0} ResNet backbones + {len(tactile_camera_names)} ViTG encoders")
    elif use_vitg:
        # Pure ViTG mode: all cameras are tactile
        backbones = None
        print("Building model with ViTG encoders only (no ResNet backbones)")
    else:
        # Pure ResNet mode: all cameras use ResNet
        backbones = []
        for _ in args.camera_names:
            backbone = build_backbone(args)
            backbones.append(backbone)

    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        # encoder = build_transformer(args)  # 错的
        encoder = build_encoder(args)  # YL change

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        vq=args.vq,
        vq_class=args.vq_class,
        vq_dim=args.vq_dim,
        action_dim=args.action_dim,
        use_vitg=use_vitg,
        vitg_ckpt_path=vitg_ckpt_path,
        tactile_camera_names=tactile_camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

