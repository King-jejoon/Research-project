"""neural_fusion_tail.py — variant of neural_fusion with TAIL-WEIGHTED convexity.

Motivation: with small global convexity weight the HI slope bends down in the
last ~10% of life (end-flattening / overfit signature).  Instead of raising the
convexity weight everywhere (which costs RMSE), weight the convexity penalty by
position in life:

    w_t = 1 + beta * (t / (T-1))**gamma          (gamma=2 default)

so mid-life stays lightly constrained (keeps the RMSE benefit) while the tail
is forced to keep accelerating.  Everything else identical to neural_fusion.
"""
import numpy as np
from neural_fusion import NeuralDataFusionModel, AdamOptimizer, delta_function


def compute_loss_and_gradients_tail(model, engine_data_list, lambda0=1, lambda1=0.001,
                                    lambda2=0.001, init_threshold=0,
                                    tail_beta=0.0, gamma=2.0,
                                    ceil_w=0.0, cmax=0.92, ceil_frac=0.90, end_target=1.0,
                                    flat_w=0.0, flat_m=0.0, flat2_w=0.0, flat2_m=0.0):
    """ceil_w > 0 adds a CEILING term: for cycles before ceil_frac of life,
    penalize HI above cmax  ->  curve cannot 'arrive early' at 1, forcing the
    steep rise into the very end (anti end-flattening)."""
    N = len(engine_data_list)
    loss_term0 = loss_term1 = loss_term2 = loss_term3 = loss_ceil = 0.0
    all_HI = []
    total_flat_acc = []
    total_flat2_acc = []

    for n in range(N):
        ed = engine_data_list[n]; T_n = len(ed)
        HI_n = model.forward(ed); all_HI.append(HI_n)

        h0 = HI_n[0, 0]
        if h0 > init_threshold:
            loss_term0 += (h0 - init_threshold) ** 2
        loss_term1 += (HI_n[-1, 0] - end_target) ** 2

        for t in range(1, T_n):
            d_nt = HI_n[t-1, 0] - HI_n[t, 0]
            loss_term2 += (1.0 / (T_n - 1)) * max(np.exp(d_nt) - 1, 0)

        for t in range(2, T_n):
            cur = HI_n[t-1, 0] - HI_n[t, 0]
            past = HI_n[t-2, 0] - HI_n[t-1, 0]
            diff = cur - past
            w_t = 1.0 + tail_beta * ((t / (T_n - 1)) ** gamma)
            loss_term3 += (w_t / (T_n - 2)) * max(np.exp(diff) - 1, 0)

        if ceil_w > 0:
            tcut = int(ceil_frac * (T_n - 1))
            for t in range(tcut):
                if HI_n[t, 0] > cmax:
                    loss_ceil += (1.0 / max(tcut, 1)) * (HI_n[t, 0] - cmax) ** 2

        if flat_w > 0 and T_n >= 10:
            a8 = int(0.8 * (T_n - 1)); a9 = int(0.9 * (T_n - 1)); aT = T_n - 1
            S9 = (HI_n[a9, 0] - HI_n[a8, 0]) / max(a9 - a8, 1)
            S10 = (HI_n[aT, 0] - HI_n[a9, 0]) / max(aT - a9, 1)
            viol = S9 - S10 + flat_m
            if viol > 0:
                total_flat_acc.append((n, viol, a8, a9, aT))

        # half-window version: slope(95-100%) must exceed slope(90-95%) + m2
        if flat2_w > 0 and T_n >= 20:
            b9 = int(0.90 * (T_n - 1)); b95 = int(0.95 * (T_n - 1)); bT = T_n - 1
            if b95 > b9 and bT > b95:
                S19 = (HI_n[b95, 0] - HI_n[b9, 0]) / (b95 - b9)
                S20 = (HI_n[bT, 0] - HI_n[b95, 0]) / (bT - b95)
                viol2 = S19 - S20 + flat2_m
                if viol2 > 0:
                    total_flat2_acc.append((n, viol2, b9, b95, bT))

    loss_flat = sum(v * v for _, v, _, _, _ in total_flat_acc)
    loss_flat2 = sum(v * v for _, v, _, _, _ in total_flat2_acc)
    total_loss = (lambda0 * loss_term0 + loss_term1 + lambda1 * loss_term2
                  + lambda2 * loss_term3 + ceil_w * loss_ceil + flat_w * loss_flat
                  + flat2_w * loss_flat2)

    total_grad_output = []
    for n in range(N):
        HI_n = all_HI[n]; T_n = len(engine_data_list[n])
        grad_h = np.zeros_like(HI_n)

        h0 = HI_n[0, 0]
        if h0 > init_threshold:
            grad_h[0, 0] += 2 * lambda0 * (h0 - init_threshold)
        grad_h[-1, 0] += 2 * (HI_n[-1, 0] - end_target)

        for t in range(1, T_n):
            d_nt = HI_n[t-1, 0] - HI_n[t, 0]
            e = np.exp(d_nt)
            g = (lambda1 / (T_n - 1)) * delta_function(e - 1) * e
            grad_h[t, 0] -= g; grad_h[t-1, 0] += g

        for t in range(2, T_n):
            cur = HI_n[t-1, 0] - HI_n[t, 0]
            past = HI_n[t-2, 0] - HI_n[t-1, 0]
            diff = cur - past
            e = np.exp(diff)
            w_t = 1.0 + tail_beta * ((t / (T_n - 1)) ** gamma)
            g = (lambda2 * w_t / (T_n - 2)) * delta_function(e - 1) * e
            grad_h[t, 0] -= g
            grad_h[t-1, 0] += 2 * g
            grad_h[t-2, 0] -= g

        if ceil_w > 0:
            tcut = int(ceil_frac * (T_n - 1))
            for t in range(tcut):
                if HI_n[t, 0] > cmax:
                    grad_h[t, 0] += ceil_w * (2.0 / max(tcut, 1)) * (HI_n[t, 0] - cmax)
        for (nn, viol, a8, a9, aT) in total_flat_acc:
            if nn != n: continue
            d9 = max(a9 - a8, 1); d10 = max(aT - a9, 1)
            g = flat_w * 2.0 * viol
            # dS9/dh: +1/d9 at a9, -1/d9 at a8 ; dS10/dh: +1/d10 at aT, -1/d10 at a9
            grad_h[a9, 0] += g * (1.0 / d9 + 1.0 / d10)
            grad_h[a8, 0] += g * (-1.0 / d9)
            grad_h[aT, 0] += g * (-1.0 / d10)
        for (nn, viol2, b9, b95, bT) in total_flat2_acc:
            if nn != n: continue
            d19 = b95 - b9; d20 = bT - b95
            g = flat2_w * 2.0 * viol2
            grad_h[b95, 0] += g * (1.0 / d19 + 1.0 / d20)
            grad_h[b9, 0] += g * (-1.0 / d19)
            grad_h[bT, 0] += g * (-1.0 / d20)
        total_grad_output.append(grad_h)

    for n in range(N):
        _ = model.forward(engine_data_list[n])
        model.backward(total_grad_output[n])
        if n == 0:
            acc = [g.copy() for g in model.gradients]
        else:
            for i in range(len(acc)): acc[i] += model.gradients[i]
    for i in range(len(acc)): acc[i] /= N
    model.gradients = acc
    return total_loss


def train_model_tail(engine_data_list, epochs=1000, lambda0=1, lambda1=0.001, lambda2=0.001,
                     init_threshold=0, tail_beta=0.0, gamma=2.0,
                     ceil_w=0.0, cmax=0.92, ceil_frac=0.90, end_target=1.0,
                     flat_w=0.0, flat_m=0.0, flat2_w=0.0, flat2_m=0.0, alpha=0.001,
                     beta1=0.9, beta2=0.999, epsilon=1e-8):
    model = NeuralDataFusionModel(input_dim=engine_data_list[0].shape[1])
    opt = AdamOptimizer(model.parameters, alpha, beta1, beta2, epsilon)
    hist = []
    for _ in range(epochs):
        loss = compute_loss_and_gradients_tail(model, engine_data_list, lambda0, lambda1,
                                               lambda2, init_threshold, tail_beta, gamma,
                                               ceil_w, cmax, ceil_frac, end_target,
                                               flat_w, flat_m, flat2_w, flat2_m)
        hist.append(loss)
        opt.step(model.parameters, model.gradients)
    return model, hist
