import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from llava.utils import rank0_print
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

try:
    from s2wrapper import forward as multiscale_forward
except:
    pass


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return
        
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        #self.device = self.vision_tower.device
        self.vision_tower.requires_grad_(False)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in range(select_every_k_layer + self.select_layer, len(image_forward_outs.hidden_states), select_every_k_layer)], dim=-1)
            select_feature_type = select_feature_type.replace("slicefour_", "")
        elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
            select_layers = [-2, -5, -8, -11, 6]
            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in select_layers], dim=-1)
            select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer] # [batch_size, 577, 1024]

        if select_feature_type == "patch":
            image_features = image_features[:, 1:]
        elif select_feature_type == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            ##### 正常走这里 #####
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype) # [batch_size, 576, 1024]
        return image_features
    
    # LX added !!!!
    def extract_features(self, images, text, low_layers, high_layers, clip_model, clip_tokenizer, T_clip, T_cls):
        # images is a tensor of shape [num_crops, 3, 336, 336]
        device = images.device
        dtype = images.dtype
        #self.vision_tower = clip_model.vision_model.to(device=device, dtype=dtype)
        outputs = self.vision_tower(images, output_hidden_states=True, output_attentions=True)
        attentions = outputs.attentions           # List[num_layers] of (B, num_heads, N, N)
        hidden_states = outputs.hidden_states     # List[num_layers + 1] of (B, N, hidden_dim)
        num_crops = images.shape[0]               # num_crops = 1 + K
        
        # -------------------------------------------------------------------------------------
        # 1. global patch importance score（只处理第0张图）
        global_att_scores = []  
        for layer in high_layers:
            attn = attentions[layer][0]           # (num_heads, N, N)
            cls2patch = attn[:, 0, 1:]            # (num_heads, 576)
            cls2patch = F.softmax(cls2patch, dim=-1)  # 每个head内softmax，(num_heads, 576)
            mean_cls2patch = cls2patch.mean(dim=0)   # [576]
            global_att_scores.append(mean_cls2patch)
        global_patch_scores = torch.stack(global_att_scores, dim=0).mean(dim=0)  # [576]   
        
        # 2. per-sub-image patch importance scores
        sub_patch_scores = []
        for i in range(1, num_crops):  # 跳过第0张（全局图）
            per_image_scores = []
            for layer in low_layers:
                attn = attentions[layer][i]           # (num_heads, N, N)
                cls2patch = attn[:, 0, 1:]            # (num_heads, 576)
                cls2patch = F.softmax(cls2patch, dim=-1)
                mean_cls2patch = cls2patch.mean(dim=0)  # [576]
                per_image_scores.append(mean_cls2patch)
            per_image_score = torch.stack(per_image_scores, dim=0).mean(dim=0)  # [576]
            sub_patch_scores.append(per_image_score)  
                      
        # 3. CLS-CLS 相似度（子图CLS vs 全局CLS）
        final_layer = -1
        # 提取所有图像的cls最终层表征
        cls_tokens = hidden_states[final_layer][:, 0]  # [num_crops, hidden_dim]
        cls_global = cls_tokens[0]                    # [hidden_dim]
        cls_sub = cls_tokens[1:]                      # [K, hidden_dim]

        cls_global_norm = normalize(cls_global.unsqueeze(0), dim=-1)   # [1, D]
        cls_sub_norm = normalize(cls_sub, dim=-1)                      # [K, D]
        similarity_scores = (cls_sub_norm @ cls_global_norm.T).squeeze(-1)  # [K]
        similarity_scores = F.softmax(similarity_scores / T_cls, dim=0)             # 归一化 [K]
        
        # 4. 子图与文本的CLIP-score
        with torch.no_grad():
            text_inputs = clip_tokenizer(text, return_tensors="pt").to(device)
            text_features = clip_model.get_text_features(**text_inputs)  # [1, D]
            text_features = normalize(text_features, dim=-1)

            image_features = []
            for i in range(1, num_crops):  # 子图1~K
                single_image = images[i].unsqueeze(0).to(device)  # [1, 3, 336, 336]
                img_feat = clip_model.get_image_features(pixel_values=single_image)  # [1, D]
                img_feat = normalize(img_feat, dim=-1)
                image_features.append(img_feat)

            image_features = torch.cat(image_features, dim=0)  # [K, D]
            clip_scores = (image_features @ text_features.T).squeeze(-1)  # [K]
            clip_scores = F.softmax(clip_scores / T_clip, dim=0)                  # 归一化
        # -------------------------------------------------------------------------------------
        # 提取原始的视觉特征
        image_features = hidden_states[self.select_layer][:, 1:] # [batch_size, 576, 1024]
        
        return image_features, global_patch_scores, sub_patch_scores, similarity_scores, clip_scores

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        _hidden_size = self.config.hidden_size
        if "slicefour" in self.select_feature:
            _hidden_size *= 4
        if "slice_m25811_f6" in self.select_feature:
            _hidden_size *= 5
        return _hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def image_size(self):
        return self.config.image_size


class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):

        self.s2_scales = getattr(args, "s2_scales", "336,672,1008")
        self.s2_scales = list(map(int, self.s2_scales.split(",")))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(vision_tower, args, delay_load)

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
            self.image_processor.size["shortest_edge"] = self.s2_image_size
            self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size["shortest_edge"] = self.s2_image_size
        self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

        self.is_loaded = True

    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)
                image_features.append(image_feature)
        else:
            image_features = multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
