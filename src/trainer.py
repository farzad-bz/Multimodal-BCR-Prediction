import torch
import torch.nn as nn
import numpy as np
from .models import SurvivalModelMM
from .losses import pairwise_ranking_loss
from lifelines.utils import concordance_index


# ======= Train one fold =======
def train_one_fold(cfg, M3D_model, modalities, ld_tr, ld_va, T_va_np, E_va_np, fold, logger, wandb_logger=None, device='cpu'):
    model = SurvivalModelMM(modalities=modalities, d_emb=cfg.fusion_model.embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=cfg.optim.scheduler_factor, patience=cfg.optim.scheduler_patience, verbose=False)
    best_c = -np.inf
    best_state = None
    bad = 0

    for epoch in range(1, cfg.train.epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0

        for step, (inputs, t, e) in enumerate(ld_tr, start=1):
            mm_inputs = {}

            if "clinical" in inputs:
                mm_inputs["clinical"] = inputs["clinical"].to(device)  # (B, 6)

            if "t2" in inputs:
                vol = inputs["t2"].to(device)  # (B, 1, D, H, W)
                with torch.no_grad():
                    if cfg.image_encoder.type == "M3D-CLIP" :
                        t2_emb = M3D_model.encode_image(vol)[:, 0]  # (B, 768)
                    else:
                        t2_emb = M3D_model.forward(vol)
                        t2_emb = torch.amax(t2_emb, dim=[2,3,4])
                mm_inputs["t2"] = t2_emb

            if "hbv" in inputs:
                vol = inputs["hbv"].to(device)
                # same preprocessing...
                with torch.no_grad():
                    if cfg.image_encoder.type == "M3D-CLIP" :
                        hbv_emb = M3D_model.encode_image(vol)[:, 0]
                    else:
                        hbv_emb = M3D_model.forward(vol)
                        hbv_emb = torch.amax(hbv_emb, dim=[2,3,4])
                mm_inputs["hbv"] = hbv_emb

            if "adc" in inputs:
                vol = inputs["adc"].to(device)
                # same preprocessing...
                with torch.no_grad():
                    if cfg.image_encoder.type == "M3D-CLIP" :
                        adc_emb = M3D_model.encode_image(vol)[:, 0]
                    else:
                        adc_emb = M3D_model.forward(vol)
                        adc_emb = torch.amax(adc_emb, dim=[2,3,4])
                mm_inputs["adc"] = adc_emb


            risk = model(mm_inputs)  # dict → risk (B,)
            loss = pairwise_ranking_loss(risk, t, e) / cfg.train.accum_steps
            loss.backward()
            running += loss.item()

            if step % cfg.train.accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)

        # ---- validation: compute risk for all val samples ----
        model.eval()
        risks = []
        with torch.no_grad():
            for inputs, t, e in ld_va:
                mm_inputs = {}

                if "clinical" in inputs:
                    mm_inputs["clinical"] = inputs["clinical"].to(device)  # (B, 6)

                if "t2" in inputs:
                    # inputs["t2"]: (B, 1, D, H, W)  ->  adapt shape to (B, 32, 256, 256)
                    vol = inputs["t2"].to(device)  # (B, 1, D, H, W)
                    with torch.no_grad():
                        if cfg.image_encoder.type == "M3D-CLIP" :
                            t2_emb = M3D_model.encode_image(vol)[:, 0]  # (B, 768)
                        else:
                            t2_emb = M3D_model.forward(vol)
                            t2_emb = torch.amax(t2_emb, dim=[2,3,4])
                    mm_inputs["t2"] = t2_emb

                if "hbv" in inputs:
                    vol = inputs["hbv"].to(device)
                    with torch.no_grad():
                        if cfg.image_encoder.type == "M3D-CLIP" :
                            hbv_emb = M3D_model.encode_image(vol)[:, 0]
                        else:
                            hbv_emb = M3D_model.forward(vol)
                            hbv_emb = torch.amax(hbv_emb, dim=[2,3,4])
                    mm_inputs["hbv"] = hbv_emb

                if "adc" in inputs:
                    vol = inputs["adc"].to(device)
                    with torch.no_grad():
                        if cfg.image_encoder.type == "M3D-CLIP" :
                            adc_emb = M3D_model.encode_image(vol)[:, 0]
                        else:
                            adc_emb = M3D_model.forward(vol)
                            adc_emb = torch.amax(adc_emb, dim=[2,3,4])
                    mm_inputs["adc"] = adc_emb

                r = model(mm_inputs).detach().cpu().numpy()
                risks.append(r)
        risk_va = np.concatenate(risks, axis=0)

        # lifelines expects higher = longer → negate risk
        c_idx = concordance_index(T_va_np, -risk_va, E_va_np)
        scheduler.step(c_idx)
        # if epoch % 10 == 0 or epoch == 1:
        logger.info(f"Epoch {epoch}: Train loss={running/len(ld_tr):.4f}, Val C-index={c_idx:.4f}")
        # early stopping
        if c_idx > best_c + 1e-4:
            best_c = c_idx
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.train.stop_patience:
                break
            
        if wandb_logger:
            lrs = [param_group['lr'] for param_group in opt.param_groups]
            wandb_logger.log({f'fold_{fold} learning_rate': sum(lrs) / len(lrs)},  step=epoch)
            wandb_logger.log({f'fold_{fold} Train Loss': running/len(ld_tr)},  step=epoch)
            wandb_logger.log({f'fold_{fold} Val C-index': c_idx},  step=epoch)

    # restore best
    if best_state is not None:
        model.load_state_dict({k: v for k, v in best_state.items()})
    return model, best_c