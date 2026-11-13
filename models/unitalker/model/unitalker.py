import os.path as osp
import torch.nn as nn

from models.unitalker.dataset.dataset_config import dataset_config
from .wav2vec2 import Wav2Vec2Model
from .base_model import BaseModel
from .wavlm import WavLMModel


class OutHead(BaseModel):
    def __init__(self, args, out_dim: int):
        super().__init__()
        head_layer = []
        in_dim = args.decoder_dimension
        for _ in range(args.headlayer - 1):
            head_layer.append(nn.Linear(in_dim, in_dim))
            head_layer.append(nn.Tanh())
        head_layer.append(nn.Linear(in_dim, out_dim))
        self.head_layer = nn.Sequential(*head_layer)

    def forward(self, x):
        return self.head_layer(x)


class UniTalker(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if "wav2vec2" in args.audio_encoder_repo:
            self.audio_encoder = Wav2Vec2Model.from_pretrained(args.audio_encoder_repo)
        elif "wavlm" in args.audio_encoder_repo:
            self.audio_encoder = WavLMModel.from_pretrained(args.audio_encoder_repo)
        else:
            raise ValueError("wrong audio_encoder_repo")
        self.audio_encoder.feature_extractor._freeze_parameters()  # conv front-end 항상 freeze
        if args.freeze_wav2vec:
            self.audio_encoder._freeze_wav2vec2_parameters()

        if args.decoder_type == "conv":
            from .unitalker_decoder import UniTalkerDecoderTCN as Decoder
        elif args.decoder_type == "transformer":
            from .unitalker_decoder import UniTalkerDecoderTransformer as Decoder
        else:
            raise ValueError("unknown decoder type")
        self.decoder = Decoder(args)

        if args.use_pca:
            pca_layer_dict = {}
            from .pca import PCALayer

            for dataset_name in args.dataset:
                if dataset_config[dataset_name]["pca"]:
                    annot_type = dataset_config[dataset_name]["annot_type"]
                    dirname = dataset_config[dataset_name]["dirname"]
                    pca_path = osp.join(args.data_root, dirname, "pca.npz")
                    pca_layer_dict[annot_type] = PCALayer(pca_path)
            self.pca_layer_dict = nn.ModuleDict(pca_layer_dict)
        else:
            self.pca_layer_dict = {}

        self.pca_dim = args.pca_dim
        self.use_pca = args.use_pca
        self.interpolate_pos = args.interpolate_pos

        out_head_dict = {}
        for dataset_name in args.dataset:
            annot_type = dataset_config[dataset_name]["annot_type"]
            out_dim = dataset_config[dataset_name]["annot_dim"]
            if args.use_pca is True and annot_type in self.pca_layer_dict:
                pca_dim = self.pca_layer_dict[annot_type].pca_dim
                out_dim = min(args.pca_dim, out_dim, pca_dim)
            if args.headlayer == 1:
                out_projection = nn.Linear(args.decoder_dimension, out_dim)
                # weight/bias=0 → 초기 pred ≈ template (residual head warm start)
                nn.init.constant_(out_projection.weight, 0)
                nn.init.constant_(out_projection.bias, 0)
            else:
                out_projection = OutHead(args, out_dim)
            out_head_dict[annot_type] = out_projection
        self.out_head_dict = nn.ModuleDict(out_head_dict)
        self.identity_num = args.identity_num

        # 옵션 헤드들(실험 코드 호환용)
        self.use_viseme_aux = getattr(args, "use_viseme_aux", False)
        if self.use_viseme_aux:
            num_viseme = getattr(args, "num_viseme_classes", 9)
            self.viseme_head = nn.Linear(args.decoder_dimension, num_viseme)
        else:
            self.viseme_head = None

    def forward(
        self,
        audio,
        template,
        face_motion,
        style_idx,
        annot_type: str,
        fps: float = None,
        viseme_labels=None,
    ):
        # face_motion(GT) 있으면 T=fps-aligned frame 수. 학습 시 data.shape[1]과 동일.
        if face_motion is not None:
            frame_num = face_motion.shape[1]
        else:
            frame_num = round(audio.shape[-1] / 16000 * fps)

        # audio [B,N] → hidden [B,T,768] (interpolate_pos=1: encoder 출력을 motion T에 맞춤)
        hidden_states = self.audio_encoder(audio, frame_num=frame_num, interpolate_pos=self.interpolate_pos)
        hidden_states = hidden_states.last_hidden_state
        # style_idx(speaker id) → decoder [B,T,decoder_dim]
        decoder_out = self.decoder(hidden_states, style_idx, frame_num)

        if (not self.use_pca) or (annot_type not in self.pca_layer_dict):
            # head: delta control. template 더하면 identity neutral + delta = absolute control
            # MetaHuman D8: decoder_out [B,T,256] → out_motion [B,T,200]
            out_motion = self.out_head_dict[annot_type](decoder_out)
            out_motion = out_motion + template
            out_pca, gt_pca = None, None
        else:
            out_pca = self.out_head_dict[annot_type](decoder_out)
            out_motion = self.pca_layer_dict[annot_type].decode(out_pca, self.pca_dim)
            out_motion = out_motion + template
            if face_motion is not None:
                gt_pca = self.pca_layer_dict[annot_type].encode(face_motion - template, self.pca_dim)
            else:
                gt_pca = None

        viseme_logits = self.viseme_head(decoder_out) if self.viseme_head is not None else None
        return out_motion, out_pca, gt_pca, viseme_logits
