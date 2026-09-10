import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptFormerBlock(nn.Module):
    """
    Single AdaptFormer bottleneck applied to spatial feature maps.
    down_proj → ReLU → up_proj → × scale → + residual
    """
    def __init__(self, channels, bottleneck=64,
                 dropout=0.1, scale=0.1):
        super().__init__()
        self.down  = nn.Conv2d(channels, bottleneck, 1)
        self.up    = nn.Conv2d(bottleneck, channels, 1)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout2d(dropout)
        self.norm  = nn.BatchNorm2d(channels)
        self.scale = scale
        nn.init.kaiming_normal_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.scale * self.up(
                   self.drop(self.act(self.down(x))))


class SceneNormalizationAdapter(nn.Module):
    """
    Block 1: Scene Normalization Adapter (AdaptFormer).

    Adapts HICO-DET images for unconstrained environments
    BEFORE feature extraction. Both YOLO and CLIP in Block 2
    receive this adapted input.

    Architecture:
      Input (B,3,224,224) — CLIP-normalized HICO-DET image
        ↓
      Conv stem (3→64 channels)
        ↓
      3 × AdaptFormer bottleneck blocks
        ↓
      Conv restore (64→3 channels)
        ↓
      Residual + LayerNorm
        ↓
      Output (B,3,224,224) — adapted image, same shape

    Output feeds BOTH:
      → YOLO11x (for human/object detection)
      → CLIP ViT-B/32 (for semantic feature extraction)
    """
    def __init__(self, bottleneck=64, dropout=0.1):
        super().__init__()
        # Stem: expand to feature channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Three AdaptFormer blocks
        self.adapt_blocks = nn.Sequential(
            AdaptFormerBlock(64, bottleneck, dropout),
            AdaptFormerBlock(64, bottleneck, dropout),
            AdaptFormerBlock(64, bottleneck, dropout),
        )
        # Restore: back to 3 channels
        self.restore = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
        )
        # Learnable scale for residual blend
        self.gate = nn.Parameter(torch.zeros(1))

        n = sum(p.numel() for p in self.parameters())
        print("="*65)
        print("Block 1 : Scene Normalization Adapter (AdaptFormer)")
        print("="*65)
        print(f"  Architecture   : AdaptFormer")
        print(f"  Adapt blocks   : 3")
        print(f"  Bottleneck dim : {bottleneck}")
        print(f"  Total params   : {n:,}  (all trainable)")
        print(f"  Input          : (B, 3, 224, 224)  HICO-DET image")
        print(f"  Output         : (B, 3, 224, 224)  adapted image")
        print(f"  Feeds into     : Block 2 (YOLO11 + CLIP ViT-B/32)")
        print("="*65)

    def forward(self, x):
        """
        x      : (B, 3, 224, 224) CLIP-normalized HICO-DET image
        return : (B, 3, 224, 224) scene-adapted image
        """
        feats   = self.stem(x)            # (B, 64, 224, 224)
        adapted = self.adapt_blocks(feats) # (B, 64, 224, 224)
        delta   = self.restore(adapted)    # (B, 3, 224, 224)
        # Residual: original + gated adaptation
        return x + torch.tanh(self.gate) * delta


# ── Verify ───────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scene_adapter = SceneNormalizationAdapter(bottleneck=64).to(device)

scene_adapter.eval()
with torch.no_grad():
    dummy_img = torch.randn(4, 3, 224, 224).to(device)
    adapted   = scene_adapter(dummy_img)

print(f"\n  Forward pass:")
print(f"  Input  : {tuple(dummy_img.shape)}")
print(f"  Output : {tuple(adapted.shape)}")
print(f"  Gate   : {scene_adapter.gate.item():.4f}"
      f"  (near 0 = near-identity at init)")
assert tuple(adapted.shape) == (4, 3, 224, 224)
print("✅ Block 1 — SceneNormalizationAdapter ready")

# --- CELL 20 (code) ---
import clip
import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLOv11FeatureExtractor(nn.Module):
    """
    YOLOv11x for human + object detection — inference ONLY.
    Never trains. Never calls .train() on YOLO internals.
    """
    def __init__(self, model_size='x', conf=0.3, device='cuda'):
        super().__init__()
        # Load weights only — no training setup triggered
        self.yolo   = YOLO(f'yolo11{model_size}.pt',
                           task='detect', verbose=False)
        self.conf   = conf
        self.device = device
        self.names  = self.yolo.names

        # Force eval and freeze all YOLO parameters immediately
        self.yolo.model.eval()
        for p in self.yolo.model.parameters():
            p.requires_grad = False

        n = sum(p.numel() for p in self.yolo.model.parameters())
        print(f"  YOLOv11x : {len(self.names)} COCO classes | "
              f"{n:,} params (frozen, inference only)")

    def train(self, mode=True):
        # Override train() to ensure YOLO model always stays in eval
        self.training = mode # Set the training flag for YOLOv11FeatureExtractor itself
        self.yolo.model.eval() # Ensure the internal PyTorch model of YOLO is always in eval mode
        return self

    def eval(self):
        # Override eval() to ensure YOLO model always stays in eval
        self.training = False # Set the training flag for YOLOv11FeatureExtractor itself
        self.yolo.model.eval() # Ensure the internal PyTorch model of YOLO is always in eval mode
        return self

    def detect(self, images_np):
        """
        images_np : list of (H,W,3) uint8 numpy arrays
        Returns   : list of detection dicts per image
        """
        self.yolo.model.eval()
        results_all = []

        for img in images_np:
            # Use predict() not __call__() — avoids training hooks
            r = self.yolo.predict(
                    img,
                    conf=self.conf,
                    verbose=False,
                    save=False,        # no file saving
                    save_txt=False,
                    save_crop=False,
                )[0]

            human_boxes, object_boxes, obj_classes = [], [], []

            if r.boxes is not None and len(r.boxes):
                boxes   = r.boxes.xyxyn.cpu()
                cls_ids = r.boxes.cls.cpu().int()
                for box, cid in zip(boxes, cls_ids):
                    name = self.names[int(cid)]
                    b    = box.tolist()
                    if name == 'person':
                        human_boxes.append(b)
                    else:
                        object_boxes.append(b)
                        obj_classes.append(name)

            results_all.append({
                'human_boxes'   : human_boxes  or [[0.0, 0.0, 0.5, 1.0]],
                'object_boxes'  : object_boxes or [[0.5, 0.0, 1.0, 1.0]],
                'object_classes': obj_classes  or ['object'],
            })
        return results_all

    def get_best_pair(self, det):
        hb = torch.tensor(det['human_boxes'][0],  dtype=torch.float16)
        ob = torch.tensor(det['object_boxes'][0], dtype=torch.float16)
        return hb, ob

    def to_uint8_numpy(self, clip_tensor):
        """CLIP-normalized tensor → uint8 numpy for YOLO."""
        mean = torch.tensor([0.48145466, 0.4578275,  0.40821073])
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        imgs = clip_tensor.cpu()
        imgs = imgs * std[None,:,None,None] + mean[None,:,None,None]
        imgs = (imgs * 255).clamp(0, 255).byte()
        return [img.permute(1,2,0).numpy() for img in imgs]


class CLIPFeatureExtractor(nn.Module):
    """
    CLIP ViT-B/32 for semantic visual feature extraction.
    Frozen — pretrained knowledge preserved.
    Output: (B, 512) semantic visual embedding.
    """
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        for p in self.visual.parameters():
            p.requires_grad = False

        n = sum(p.numel() for p in self.visual.parameters())
        print(f"  CLIP ViT-B/32 : {n:,} params (frozen)")

    def forward(self, x):
        """x: (B, 3, 224, 224) → (B, 512)"""
        vit = self.visual
        x   = vit.conv1(x)
        x   = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1)
        cls = vit.class_embedding.unsqueeze(0).expand(x.shape[0],-1,-1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + vit.positional_embedding
        x   = vit.ln_pre(x)
        x   = x.permute(1,0,2)
        x   = vit.transformer(x)
        x   = x.permute(1,0,2)
        x   = vit.ln_post(x[:,0,:])
        if vit.proj is not None:
            x = x @ vit.proj
        return x   # (B, 512)


class ImageFeatureExtraction(nn.Module):
    """
    Block 2: Image Feature Extraction.
    YOLOv11x (inference only) + CLIP ViT-B/32 (frozen).

    At TRAINING : GT boxes from HICO-DET used directly.
                  YOLO does NOT run at training time.
    At INFERENCE: YOLO detects boxes from adapted image.
                  CLIP extracts semantic features always.
    """
    def __init__(self, clip_model, yolo_size='x',
                 conf=0.3, device='cuda'):
        super().__init__()
        self.clip_extractor = CLIPFeatureExtractor(clip_model)
        self.yolo_detector  = YOLOv11FeatureExtractor(
                                yolo_size, conf, device)
        self.device = device

        print("="*65)
        print("Block 2 : Image Feature Extraction")
        print("="*65)
        print(f"  YOLOv11x       : detection only — NEVER trains")
        print(f"  CLIP ViT-B/32  : semantic features — frozen")
        print(f"  Input          : adapted image (B,3,224,224)")
        print(f"  Output clip    : (B,512) semantic embedding")
        print(f"  Output boxes   : (B,4) human + (B,4) object [0-1]")
        print(f"  Training       : GT boxes used, YOLO skipped")
        print(f"  Inference      : YOLO detects, CLIP encodes")
        print("="*65)

    def forward(self, adapted_image, human_box=None,
                object_box=None, use_gt_boxes=True):
        # CLIP semantic features — always runs
        clip_feat = self.clip_extractor(adapted_image)   # (B,512)

        # YOLO spatial detection — inference only
        # At training: GT boxes passed in directly
        if not use_gt_boxes:
            imgs_np    = self.yolo_detector.to_uint8_numpy(adapted_image)
            detections = self.yolo_detector.detect(imgs_np)
            human_box  = torch.stack([
                self.yolo_detector.get_best_pair(d)[0]
                for d in detections]).to(self.device)
            object_box = torch.stack([
                self.yolo_detector.get_best_pair(d)[1]
                for d in detections]).to(self.device)

        return clip_feat, human_box, object_box


# ── Build and verify ─────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'

clip_model_raw, _ = clip.load("ViT-B/32", device=device)
clip_model_raw    = clip_model_raw.float()

feature_extractor = ImageFeatureExtraction(
    clip_model_raw, yolo_size='x', device=device).to(device)

# ── Confirm YOLO is NOT training ─────────────────────────
yolo_training = feature_extractor.yolo_detector.yolo.model.training
clip_training = any(p.requires_grad for p in
                    feature_extractor.clip_extractor.parameters())
yolo_grads    = any(p.requires_grad for p in
                    feature_extractor.yolo_detector.yolo.model.parameters())

print(f"\nStatus checks:")
print(f"  {'✅' if not yolo_training else '❌'}"
      f"  YOLO model.training = {yolo_training}  (should be False)")
print(f"  {'✅' if not yolo_grads else '❌'}"
      f"  YOLO requires_grad  = {yolo_grads}  (should be False)")
print(f"  {'✅' if not clip_training else '❌'}"
      f"  CLIP requires_grad  = {clip_training}  (should be False)")

# ── Forward pass check ────────────────────────────────────
feature_extractor.eval()
with torch.no_grad():
    dummy_adapted = torch.randn(4, 3, 224, 224).to(device)
    dummy_hb      = torch.rand(4, 4).to(device)
    dummy_ob      = torch.rand(4, 4).to(device)
    clip_out, hb_out, ob_out = feature_extractor(
        dummy_adapted, dummy_hb, dummy_ob,
        use_gt_boxes=True)    # ← GT boxes, YOLO skipped

print(f"\nForward pass (training mode with GT boxes):")
print(f"  Adapted image in : {tuple(dummy_adapted.shape)}")
print(f"  CLIP features    : {tuple(clip_out.shape)}")
print(f"  Human box out    : {tuple(hb_out.shape)}")
print(f"  Object box out   : {tuple(ob_out.shape)}")

checks = {
    "CLIP output (4,512)"  : tuple(clip_out.shape) == (4,512),
    "Human box (4,4)"      : tuple(hb_out.shape)   == (4,4),
    "Object box (4,4)"     : tuple(ob_out.shape)    == (4,4),
    "YOLO not training"    : not yolo_training,
    "YOLO frozen"          : not yolo_grads,
    "CLIP frozen"          : not clip_training,
}
print(f"\nVerification:")
all_ok = True
for check, result in checks.items():
    icon = "✅" if result else "❌"
    print(f"  {icon}  {check}")
    if not result: all_ok = False

print(f"\n{'✅ Block 2 ready — proceed to Cell 4' if all_ok else '❌ Fix errors above'}")


# --- CELL 21 (code) ---
# ============================================================
# CELL 4: Block 3 — Multimodal Feature Fusion
# Architecture: Cross-attention transformer
# Inputs:
#   → CLIP features (B,512) from Block 2
#   → Human box    (B,4)    from Block 2
#   → Object box   (B,4)    from Block 2
# Output: Fused multimodal features (B,512)
# ============================================================
import torch
import torch.nn as nn


class SpatialEncoder(nn.Module):
    """Encodes box geometry into 512-dim spatial tokens."""
    def __init__(self, dim=512):
        super().__init__()
        self.human_enc = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(),
            nn.Linear(128, dim), nn.LayerNorm(dim))
        self.obj_enc = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(),
            nn.Linear(128, dim), nn.LayerNorm(dim))
        self.rel_enc = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, dim), nn.LayerNorm(dim))

    def _box8(self, b):
        x1,y1,x2,y2 = b[:,0],b[:,1],b[:,2],b[:,3]
        return torch.stack([x1,y1,x2,y2,
                            (x1+x2)/2,(y1+y2)/2,
                            x2-x1,y2-y1], dim=-1)

    def _rel(self, hb, ob):
        hcx=(hb[:,0]+hb[:,2])/2; hcy=(hb[:,1]+hb[:,3])/2
        ocx=(ob[:,0]+ob[:,2])/2; ocy=(ob[:,1]+ob[:,3])/2
        dx=ocx-hcx; dy=ocy-hcy
        dist=torch.sqrt(dx**2+dy**2+1e-8)
        ha=(hb[:,2]-hb[:,0])*(hb[:,3]-hb[:,1])
        oa=(ob[:,2]-ob[:,0])*(ob[:,3]-ob[:,1])
        return torch.stack([dx,dy,dist,ha-oa],dim=-1)

    def forward(self, hb, ob):
        return torch.stack([
            self.human_enc(self._box8(hb)),
            self.obj_enc(  self._box8(ob)),
            self.rel_enc(  self._rel(hb,ob)),
        ], dim=1)   # (B, 3, 512)


class CrossAttnBlock(nn.Module):
    def __init__(self, dim=512, heads=8, dropout=0.1):
        super().__init__()
        self.nq   = nn.LayerNorm(dim)
        self.nkv  = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
                      dim, heads, dropout=dropout, batch_first=True)
        self.nff  = nn.LayerNorm(dim)
        self.ff   = nn.Sequential(
            nn.Linear(dim,dim*4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4,dim), nn.Dropout(dropout))

    def forward(self, q, kv):
        a, _ = self.attn(self.nq(q), self.nkv(kv), self.nkv(kv))
        q    = q + a
        return q + self.ff(self.nff(q))


class MultimodalFeatureFusion(nn.Module):
    """
    Block 3: Multimodal Feature Fusion.
    Cross-attention transformer.
    Fuses CLIP semantic features with YOLOv11 spatial features.
    Output: (B, 512)
    """
    def __init__(self, dim=512, heads=8, layers=3, dropout=0.1):
        super().__init__()
        self.spatial = SpatialEncoder(dim)
        self.v2s = nn.ModuleList([
            CrossAttnBlock(dim,heads,dropout) for _ in range(layers)])
        self.s2v = nn.ModuleList([
            CrossAttnBlock(dim,heads,dropout) for _ in range(layers)])
        self.proj = nn.Sequential(
            nn.Linear(dim*2,dim), nn.LayerNorm(dim), nn.GELU())

        n = sum(p.numel() for p in self.parameters())
        print("="*65)
        print("Block 3 : Multimodal Feature Fusion")
        print("="*65)
        print(f"  Architecture   : Cross-attention transformer")
        print(f"  Attention layers : {layers}")
        print(f"  Attention heads  : {heads}")
        print(f"  Total params     : {n:,}")
        print(f"  Input CLIP     : (B,512) from Block 2")
        print(f"  Input boxes    : (B,4)+(B,4) from Block 2")
        print(f"  Output         : (B,512) fused features")
        print("="*65)

    def forward(self, clip_feat, human_box, object_box):
        sp = self.spatial(human_box, object_box)  # (B,3,512)
        v  = clip_feat.unsqueeze(1)               # (B,1,512)
        for v2s, s2v in zip(self.v2s, self.s2v):
            v  = v2s(v,  sp)
            sp = s2v(sp, v)
        return self.proj(
            torch.cat([v.squeeze(1), sp.mean(1)], dim=-1))


# ── Verify ───────────────────────────────────────────────
fusion = MultimodalFeatureFusion(dim=512, heads=8, layers=3).to(device)

fusion.eval()
with torch.no_grad():
    dummy_clip = torch.randn(4, 512).to(device)
    dummy_hb   = torch.rand(4, 4).to(device)
    dummy_ob   = torch.rand(4, 4).to(device)
    fused_out  = fusion(dummy_clip, dummy_hb, dummy_ob)

print(f"\n  Forward pass:")
print(f"  CLIP in    : {tuple(dummy_clip.shape)}")
print(f"  Boxes in   : {tuple(dummy_hb.shape)} + {tuple(dummy_ob.shape)}")
print(f"  Fused out  : {tuple(fused_out.shape)}")
print("✅ Block 3 — MultimodalFeatureFusion ready")

# --- CELL 22 (code) ---
# ============================================================
# CELL 5: Block 4 — HOI Recognition Module
# Architecture: RLIP Transformer
# Input : Fused features (B,512) from Block 3
# Output: HOI logits (B,600), verb (B,117), obj (B,80), conf (B,1)
# ============================================================
import torch
import torch.nn as nn


class HOIRecognitionModule(nn.Module):
    """
    Block 4: HOI Recognition Module (RLIP Transformer).
    Predicts HOI triplets from fused multimodal features.
    Output feeds into Block 5 (Interaction Prior Database).
    """
    def __init__(self, dim=512, num_hoi=600,
                 num_verb=117, num_obj=80, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=8, dim_feedforward=2048,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=2)

        self.hoi_head = nn.Sequential(
            nn.Linear(dim,512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512,num_hoi))
        self.verb_head = nn.Sequential(
            nn.Linear(dim,256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256,num_verb))
        self.obj_head = nn.Sequential(
            nn.Linear(dim,256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256,num_obj))
        self.conf_head = nn.Sequential(
            nn.Linear(dim,64), nn.ReLU(),
            nn.Linear(64,1), nn.Sigmoid())

        n = sum(p.numel() for p in self.parameters())
        print("="*65)
        print("Block 4 : HOI Recognition Module (RLIP Transformer)")
        print("="*65)
        print(f"  Architecture  : RLIP Transformer")
        print(f"  Encoder layers: 2")
        print(f"  Attention heads: 8")
        print(f"  Total params  : {n:,}")
        print(f"  Input         : (B,512) fused features")
        print(f"  Output HOI    : (B,{num_hoi}) HOI logits")
        print(f"  Output verb   : (B,{num_verb}) verb logits")
        print(f"  Output obj    : (B,{num_obj}) object logits")
        print(f"  Output conf   : (B,1) confidence")
        print(f"  Feeds into    : Block 5 (Interaction Prior Database)")
        print("="*65)

    def forward(self, fused_feat):
        x = fused_feat.unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return {
            'hoi' : self.hoi_head(x),    # (B, 600)
            'verb': self.verb_head(x),   # (B, 117)
            'obj' : self.obj_head(x),    # (B, 80)
            'conf': self.conf_head(x),   # (B, 1)
            'feat': x,                   # (B, 512)
        }


# ── Verify ───────────────────────────────────────────────
hoi_recognition = HOIRecognitionModule(
    dim=512, num_hoi=600, num_verb=117, num_obj=80).to(device)

hoi_recognition.eval()
with torch.no_grad():
    dummy_fused = torch.randn(4, 512).to(device)
    rlip_out    = hoi_recognition(dummy_fused)

print(f"\n  Forward pass:")
print(f"  Input  : {tuple(dummy_fused.shape)}")
for k, v in rlip_out.items():
    print(f"  {k:<6} : {tuple(v.shape)}")
print("✅ Block 4 — HOIRecognitionModule ready")

# --- CELL 23 (code) ---
# ============================================================
# CELL 6: Block 5 — Interaction Prior Database
# Architecture: Learnable embedding table (600 × 512)
#               + alignment head + re-ranking layer
#
# Position in pipeline: BETWEEN RLIP and LLM Reasoning.
# Takes RLIP raw predictions → re-ranks → refined predictions.
#
# Built offline (Cell 0A) using:
#   Qwen2.5-7B descriptions → CLIP text embeddings → 600×512 matrix
# Frozen during training.
# ============================================================
import torch
import torch.nn as nn


class InteractionPriorDatabase(nn.Module):
    """
    Block 5: Interaction Prior Database.

    Sequential position in your diagram:
      HOI Recognition (RLIP) → Interaction Prior DB → LLM Reasoning

    Three roles:
      1. Stores 600 prior embeddings (one per HOI class)
         built from Qwen2.5-7B descriptions + CLIP text encoder
      2. Aligns RLIP predictions against stored priors
      3. Re-ranks top predictions, outputs refined logits

    During training: prior weights FROZEN
    """
    def __init__(self, num_hoi=600, dim=512, dropout=0.1):
        super().__init__()
        # 600-class prior embedding table
        self.prior_emb = nn.Embedding(num_hoi, dim)
        nn.init.uniform_(self.prior_emb.weight, -0.1, 0.1)

        # Alignment: fused visual features vs prior embeddings
        self.align_head = nn.Sequential(
            nn.Linear(dim*2, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 1))

        # Plausibility: per-HOI visual plausibility score
        self.plausibility = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(),
            nn.Linear(256, num_hoi), nn.Sigmoid())

        # Blend gate: how much to trust prior vs raw RLIP
        self.blend_gate = nn.Sequential(
            nn.Linear(dim, 1), nn.Sigmoid())

        # Final re-ranking classifier
        self.reranker = nn.Sequential(
            nn.Linear(dim*2, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, num_hoi))

        n_emb   = sum(p.numel() for p in self.prior_emb.parameters())
        n_total = sum(p.numel() for p in self.parameters())
        print("="*65)
        print("Block 5 : Interaction Prior Database")
        print("="*65)
        print(f"  Prior embeddings : 600 × 512 = {n_emb:,} params")
        print(f"  Total params     : {n_total:,}")
        print(f"  Status           : frozen after injection")
        print(f"  Built from       : Qwen2.5-7B + CLIP text encoder")
        print(f"  Input            : (B,600) RLIP logits + (B,512) fused feat")
        print(f"  Output refined   : (B,600) prior-aligned HOI logits")
        print(f"  Output plaus     : (B,600) plausibility scores")
        print(f"  Feeds into       : Block 6 (Offline LLM Reasoning)")
        print("="*65)

    def get_all_priors(self):
        return self.prior_emb.weight   # (600, 512)

    def forward(self, fused_feat, rlip_hoi_logits):
        """
        fused_feat       : (B,512) from Block 3 Multimodal Fusion
        rlip_hoi_logits  : (B,600) from Block 4 HOI Recognition

        Returns:
          refined_logits  : (B,600) prior-aligned HOI predictions
          plaus_scores    : (B,600) plausibility per HOI
        """
        # Top-10 RLIP candidates
        top_ids = rlip_hoi_logits.topk(10, dim=-1).indices   # (B,10)
        priors  = self.prior_emb(top_ids)                    # (B,10,512)

        # Align visual features against each prior
        vis_exp  = fused_feat.unsqueeze(1).expand(-1,10,-1)
        align_sc = self.align_head(
            torch.cat([vis_exp, priors], dim=-1)).squeeze(-1) # (B,10)

        # Boost logits for prior-aligned candidates
        boost = torch.zeros_like(rlip_hoi_logits)
        boost.scatter_(1, top_ids, torch.sigmoid(align_sc))
        gate    = self.blend_gate(fused_feat)
        refined = rlip_hoi_logits + gate * boost              # (B,600)

        # Visual plausibility scores
        plaus = self.plausibility(fused_feat)                 # (B,600)

        # Final re-ranking using weighted prior context
        soft_att  = torch.softmax(refined, dim=-1)
        wtd_prior = soft_att @ self.get_all_priors()          # (B,512)
        combined  = torch.cat([fused_feat, wtd_prior], dim=-1)
        reranked  = self.reranker(combined)                   # (B,600)

        final_logits = reranked * plaus + boost               # (B,600)

        return {
            'refined_logits': final_logits,  # (B,600) → to Block 6
            'raw_logits'    : rlip_hoi_logits, # (B,600) → for loss
            'plaus_scores'  : plaus,           # (B,600)
        }


# ── Verify ───────────────────────────────────────────────
prior_db = InteractionPriorDatabase(
    num_hoi=600, dim=512).to(device)

prior_db.eval()
with torch.no_grad():
    dummy_fused = torch.randn(4, 512).to(device)
    dummy_hoi   = torch.randn(4, 600).to(device)
    prior_out   = prior_db(dummy_fused, dummy_hoi)

print(f"\n  Forward pass:")
print(f"  Fused feat in    : {tuple(dummy_fused.shape)}")
print(f"  RLIP logits in   : {tuple(dummy_hoi.shape)}")
print(f"  Refined logits   : {tuple(prior_out['refined_logits'].shape)}")
print(f"  Plausibility     : {tuple(prior_out['plaus_scores'].shape)}")
print("✅ Block 5 — InteractionPriorDatabase ready")

# --- CELL 24 (code) ---
import torch
import torch.nn as nn


class OfflineLLMReasoning(nn.Module):
    """
    Block 6: Offline LLM Reasoning (Qwen2.5-7B-Instruct).

    Two components:
      A) Neural commonsense verifier (trainable)
         — runs during training and inference
      B) Qwen2.5-7B (inference only, not trained)
         — loaded separately via load_qwen()

    Produces both final outputs from your diagram:
      → Final HOI Prediction
      → Natural Language Explanation
    """
    def __init__(self, dim=512, num_hoi=600, dropout=0.1):
        super().__init__()
        # Trainable neural verifier
        self.final_classifier = nn.Sequential(
            nn.Linear(dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, num_hoi))
        self.confidence_head = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(),
            nn.Linear(64, 1)) # Removed nn.Sigmoid()

        self.qwen = None   # loaded via load_qwen()

        n = sum(p.numel() for p in self.parameters())
        print("="*65)
        print("Block 6 : Offline LLM Reasoning")
        print("="*65)
        print(f"  LLM              : Qwen2.5-7B-Instruct (offline)")
        print(f"  Neural verifier  : {n:,} params (trainable)")
        print(f"  Input            : (B,600) from Block 5")
        print(f"  Output 1         : Final HOI Prediction")
        print(f"  Output 2         : Natural Language Explanation")
        print(f"  Qwen2.5 status   : call load_qwen() at inference")
        print("="*65)

    def load_qwen(self, use_4bit=True):
        """Load Qwen2.5-7B for commonsense grounding + NL generation."""
        from transformers import (AutoTokenizer,
                                  AutoModelForCausalLM,
                                  BitsAndBytesConfig)
        MODEL = "Qwen/Qwen2.5-7B-Instruct"
        print(f"Loading {MODEL} ({'4-bit' if use_4bit else 'full'})...")
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ) if use_4bit else None
        self.tok = AutoTokenizer.from_pretrained(
            MODEL, trust_remote_code=True)
        self.qwen = AutoModelForCausalLM.from_pretrained(
            MODEL, quantization_config=quant,
            device_map="auto", torch_dtype=torch.float16,
            trust_remote_code=True)
        self.qwen.eval()
        print("✅ Qwen2.5-7B-Instruct loaded")

    def _chat(self, user_msg, max_tokens=80):
        msgs = [{"role": "system",
                 "content": "You are an expert in visual commonsense "
                            "reasoning for human-object interaction "
                            "detection. Be concise and precise."},
                {"role": "user", "content": user_msg}]
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.qwen.device)
        with torch.no_grad():
            out = self.qwen.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=0.3, do_sample=True,
                pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(
            out[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True).strip()

    def score_commonsense(self, verb, obj, context=""):
        """
        Qwen2.5 scores HOI commonsense plausibility.
        Used to re-rank top-3 neural predictions → Final HOI Prediction.
        """
        v, o = verb.replace('_',' '), obj.replace('_',' ')
        msg  = (f'Rate how plausible it is that a person would be '
                f'"{v}" a/an "{o}" in a real scene.\n'
                f'Context: {context or "general"}\n'
                f'Reply ONLY with a decimal number 0.0 to 1.0.')
        raw = self._chat(msg, max_tokens=10)
        try:
            import re
            num = re.search(r'0?\.\d+|1\.0|0|1', raw)
            return float(num.group()) if num else 0.5
        except Exception:
            return 0.5

    def generate_nl_explanation(self, verb, obj, conf, context=""):
        """
        Qwen2.5 generates Natural Language Explanation.
        Final output from the bottom of your diagram.
        """
        v, o = verb.replace('_',' '), obj.replace('_',' ')
        an   = 'an' if o[0] in 'aeiou' else 'a'
        msg  = (f'A computer vision model detected:\n'
                f'  Action    : person is "{v}" {an} "{o}"\n'
                f'  Confidence: {conf:.0%}\n'
                f'  Scene     : {context or "HICO-DET unconstrained"}\n\n'
                f'Write a natural 2-sentence explanation describing '
                f'what the person is doing and their spatial '
                f'relationship with the object.')
        return self._chat(msg, max_tokens=80)

    def build_hoi_descriptions(self, hoi_list):
        """
        Generates Qwen2.5 descriptions for all 600 HOIs.
        Called ONCE in Cell 0A to build Interaction Prior Database.
        """
        descriptions = {}
        for i, hoi in enumerate(hoi_list):
            v  = hoi['verb'].replace('_',' ')
            o  = hoi['object'].replace('_',' ')
            an = 'an' if o[0] in 'aeiou' else 'a'
            msg = (f'Describe in ONE sentence what it visually '
                   f'looks like when a person is "{v}" {an} "{o}". '
                   f'Focus on body position and spatial layout.')
            desc = self._chat(msg, max_tokens=60)
            descriptions[hoi['hoi_id']] = {
                'verb': hoi['verb'], 'object': hoi['object'],
                'desc': desc, 'model': 'Qwen2.5-7B-Instruct'}
            if i % 50 == 0:
                print(f"  {i+1:3d}/600: {v:<20} {o:<15}"
                      f"→ {desc[:45]}...")
        return descriptions

    def forward(self, fused_feat, refined_logits):
        """
        Neural forward — runs during BOTH training and inference.
        No Qwen2.5 call here — fast and differentiable.
        fused_feat     : (B,512)
        refined_logits : (B,600) from Block 5
        """
        # Combine fused visual with logit-weighted prior context
        final_logits = self.final_classifier(fused_feat)  # (B,600)
        confidence   = self.confidence_head(fused_feat)   # (B,1)
        # Element-wise grounding with prior-refined logits
        grounded     = final_logits + refined_logits       # (B,600)
        return {
            'final_hoi'  : grounded,     # (B,600) → Final HOI Prediction
            'confidence' : confidence,   # (B,1)
        }

    def get_final_outputs(self, final_hoi, hoi_list,
                          detected_objects=None):
        """
        Full Block 6 inference producing BOTH diagram outputs.
        Requires load_qwen() first.
        """
        assert self.qwen is not None, \
               "Call model.llm_reasoning.load_qwen() first"

        probs   = torch.softmax(final_hoi, dim=-1)
        context = (f"Objects: {', '.join(detected_objects)}"
                   if detected_objects else "")
        results = []

        for b in range(final_hoi.shape[0]):
            top3_ids  = probs[b].topk(3).indices.tolist()
            top3_conf = probs[b].topk(3).values.tolist()

            # Final HOI Prediction — Qwen2.5 commonsense re-ranking
            best = {'score': -1}
            for hoi_idx, conf in zip(top3_ids, top3_conf):
                hoi   = hoi_list[hoi_idx]
                cs_sc = self.score_commonsense(
                            hoi['verb'], hoi['object'], context)
                combined = 0.7*conf + 0.3*cs_sc
                if combined > best['score']:
                    best = {
                        'score': combined, 'hoi_idx': hoi_idx,
                        'verb': hoi['verb'], 'object': hoi['object'],
                        'hoi_id': hoi['hoi_id'],
                        'neural_conf': conf, 'cs_score': cs_sc}

            # Natural Language Explanation
            nl_expl = self.generate_nl_explanation(
                          best['verb'], best['object'],
                          best['neural_conf'], context)
            results.append({
                'hoi_id'        : best['hoi_id'],
                'verb'          : best['verb'],
                'object'        : best['object'],
                'neural_conf'   : best['neural_conf'],
                'cs_score'      : best['cs_score'],
                'combined_score': best['score'],
                'nl_explanation': nl_expl,
            })
        return results


# ── Verify ───────────────────────────────────────────────
llm_reasoning = OfflineLLMReasoning(dim=512, num_hoi=600).to(device)

llm_reasoning.eval()
with torch.no_grad():
    dummy_fused   = torch.randn(4, 512).to(device)
    dummy_refined = torch.randn(4, 600).to(device)
    llm_out       = llm_reasoning(dummy_fused, dummy_refined)

print(f"\n  Forward pass:")
print(f"  Fused feat in    : {tuple(dummy_fused.shape)}")
print(f"  Refined logits   : {tuple(dummy_refined.shape)}")
print(f"  Final HOI out    : {tuple(llm_out['final_hoi'].shape)}")
print(f"  Confidence out   : {tuple(llm_out['confidence'].shape)}")
print("✅ Block 6 — OfflineLLMReasoning ready")

# --- CELL 25 (markdown) ---
## Load the precomputed Interaction Prior Database (Kaggle input)

# --- CELL 26 (code) ---
# ============================================================
# CELL 0A: Load Precomputed Interaction Prior Database
# (Colab loaded this from Google Drive; here it comes straight
#  was auto-detected in Cell 0B above)
# ============================================================
import torch



print("="*55)
print(" Interaction Prior Database Loaded")
print("="*55)
print("="*55)


# --- CELL 27 (markdown) ---
## Full HOIDetector assembly

# --- CELL 28 (code) ---
# ============================================================
# CELL 8: Full HOIDetector — Complete Architecture
# Assembles all 6 blocks matching your diagram exactly.
# ============================================================
import clip, json


class HOIDetector(nn.Module):
    """
    Complete architecture matching your diagram:

    Image Input (HICO-DET)
           ↓
    Block 1: Scene Normalization Adapter (AdaptFormer)
           ↓
    Block 2: Image Feature Extraction (YOLOv11 + CLIP ViT-B/32)
           ↓
    Block 3: Multimodal Feature Fusion (Cross-attention)
           ↓
    Block 4: HOI Recognition Module (RLIP Transformer)
           ↓
    Block 5: Interaction Prior Database
           ↓
    Block 6: Offline LLM Reasoning (Qwen2.5-7B)
           ↓               ↓
    Final HOI Prediction   Natural Language Explanation
    """
    def __init__(self, num_hoi=600, num_verb=117, num_obj=80,
                 fusion_layers=3, device='cuda'):
        super().__init__()
        self.device  = device
        self.num_hoi = num_hoi

        clip_model, _ = clip.load("ViT-B/32", device='cpu')
        clip_model     = clip_model.float()

        print("Assembling architecture blocks:")
        print("-"*65)

        # Block 1 — Scene Normalization Adapter
        self.scene_adapter = SceneNormalizationAdapter(
                               bottleneck=64)

        # Block 2 — Image Feature Extraction
        self.feature_extractor = ImageFeatureExtraction(
                                   clip_model, yolo_size='x',
                                   device=device)

        # Block 3 — Multimodal Feature Fusion
        self.fusion = MultimodalFeatureFusion(
                        dim=512, heads=8, layers=fusion_layers)

        # Block 4 — HOI Recognition Module
        self.hoi_recognition = HOIRecognitionModule(
                                 dim=512, num_hoi=num_hoi,
                                 num_verb=num_verb, num_obj=num_obj)

        # Block 5 — Interaction Prior Database
        self.prior_db = InteractionPriorDatabase(
                          num_hoi=num_hoi, dim=512)

        # Block 6 — Offline LLM Reasoning
        self.llm_reasoning = OfflineLLMReasoning(
                               dim=512, num_hoi=num_hoi)

        self.to(device)
        self._print_summary()

    def inject_priors(self, prior_matrix):
        """
        Load Qwen2.5+CLIP prior embeddings into Block 5.
        Freezes prior weights — Qwen2.5 knowledge preserved.
        """
        with torch.no_grad():
            self.prior_db.prior_emb.weight.copy_(
                prior_matrix.to(self.device))
        for p in self.prior_db.prior_emb.parameters():
            p.requires_grad = False

    def _print_summary(self):
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*65}")
        print(f"  HOIDetector — Full Architecture")
        print(f"{'='*65}")
        print(f"  Image Input (HICO-DET)")
        print(f"         ↓")
        print(f"  Block 1: Scene Normalization Adapter (AdaptFormer)")
        print(f"         ↓")
        print(f"  Block 2: Image Feature Extraction (YOLOv11 + CLIP)")
        print(f"         ↓")
        print(f"  Block 3: Multimodal Feature Fusion (Cross-attention)")
        print(f"         ↓")
        print(f"  Block 4: HOI Recognition Module (RLIP Transformer)")
        print(f"         ↓")
        print(f"  Block 5: Interaction Prior Database")
        print(f"         ↓")
        print(f"  Block 6: Offline LLM Reasoning (Qwen2.5-7B)")
        print(f"         ↓               ↓")
        print(f"  Final HOI Prediction   Natural Language Explanation")
        print(f"{'='*65}")
        print(f"  Trainable  : {trainable:,}")
        print(f"  Frozen     : {total-trainable:,}  (CLIP ViT-B/32)")
        print(f"  Total      : {total:,}")
        print(f"{'='*65}\n")

    def forward(self, image, human_box=None, object_box=None,
                use_gt_boxes=True):
        """
        image         : (B, 3, 224, 224) HICO-DET image
        human_box     : (B, 4) GT or YOLOv11 detected
        object_box    : (B, 4) GT or YOLOv11 detected
        use_gt_boxes  : True at training, False at inference
        """
        # Block 1 — Scene Normalization Adapter
        adapted = self.scene_adapter(image)             # (B,3,224,224)

        # Block 2 — Image Feature Extraction (YOLOv11 + CLIP)
        clip_feat, human_box, object_box = \
            self.feature_extractor(
                adapted, human_box, object_box, use_gt_boxes)

        # Block 3 — Multimodal Feature Fusion
        fused = self.fusion(
                    clip_feat, human_box, object_box)   # (B,512)

        # Block 4 — HOI Recognition Module (RLIP)
        rlip_out = self.hoi_recognition(fused)

        # Block 5 — Interaction Prior Database
        prior_out = self.prior_db(fused, rlip_out['hoi'])

        # Block 6 — Offline LLM Reasoning (neural part)
        llm_out = self.llm_reasoning(
                      fused, prior_out['refined_logits'])

        return {
            # Final outputs (bottom of diagram)
            'final_hoi'     : llm_out['final_hoi'],       # (B,600)
            # Auxiliary outputs for loss computation
            'rlip_logits'   : rlip_out['hoi'],            # (B,600)
            'prior_logits'  : prior_out['refined_logits'],# (B,600)
            'prior_raw'     : prior_out['raw_logits'],    # (B,600)
            'verb_logits'   : rlip_out['verb'],           # (B,117)
            'obj_logits'    : rlip_out['obj'],            # (B,80)
            'confidence'    : llm_out['confidence'],      # (B,1)
            'plaus_scores'  : prior_out['plaus_scores'],  # (B,600)
            'fused_feat'    : fused,                      # (B,512)
        }


# ── Build model ───────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
