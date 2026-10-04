# -*- coding: utf-8 -*-
"""Yamaguchi four-component decomposition (mode 0, deorientation only), ported
line by line from the group's own MATLAB code:

  D:\\雷达\\数据库\\yamaguchi_4components_T3.m   (Sinong Quan, 2016)
  D:\\雷达\\数据库\\unitary_rotation.m
  D:\\雷达\\数据库\\C3_T3.m

mode 0 = deorientation by the polarization orientation angle, HV_type fixed to
1 (volume + double bounce dominant), then the Yamaguchi four-component branch
with the Freeman-Yamaguchi three-component fallback when the volume power is
negative.  Phases/outputs follow the MATLAB exactly; Podd = Ps, Pdbl + Phel =
Pm, Pvol = Pv, matching this paper's Pm/Pv/Ps.
"""
import numpy as np

EPS = 1e-15


def c3_to_t3(C11, C12, C13, C22, C23, C33):
    """Covariance -> coherency, exactly as C3_T3.m (inputs complex or real)."""
    T11 = 0.5 * (C11 + C33 + 2 * C13.real)
    T12 = (C11 - C33) / 2 + 1j * (-C13.imag)
    T13 = (C12.real + C23.real) / np.sqrt(2) + 1j * (C12.imag - C23.imag) / np.sqrt(2)
    T22 = 0.5 * (C11 + C33 - 2 * C13.real)
    T23 = (C12.real - C23.real) / np.sqrt(2) + 1j * (C12.imag + C23.imag) / np.sqrt(2)
    T33 = C22
    return T11, T12, T13, T22, T23, T33


def unitary_rotation(T11, T12, T13, T22, T23, T33, teta):
    """unitary_rotation.m, vectorised."""
    c, s = np.cos(teta), np.sin(teta)
    DT11 = T11
    DT12 = (T12.real * c + T13.real * s) + 1j * (T12.imag * c + T13.imag * s)
    DT13 = (-T12.real * s + T13.real * c) + 1j * (-T12.imag * s + T13.imag * c)
    DT22 = T22 * c * c + 2 * T23.real * c * s + T33 * s * s
    DT23_re = -T22 * c * s + T23.real * c * c - T23.real * s * s + T33 * c * s
    DT23_im = T23.imag * c * c + T23.imag * s * s
    DT33 = T22 * s * s + T33 * c * c - 2 * T23.real * c * s
    return DT11, DT12, DT13, DT22, DT23_re + 1j * DT23_im, DT33


def yamaguchi4_mode0(T11, T12, T13, T22, T23, T33):
    """Returns (Podd, Pdbl, Pvol, Phel) in the input's units."""
    T11 = np.asarray(T11, np.float64)
    T22 = np.asarray(T22, np.float64)
    T33 = np.asarray(T33, np.float64)
    T12 = np.asarray(T12, np.complex128)
    T13 = np.asarray(T13, np.complex128)
    T23 = np.asarray(T23, np.complex128)

    Phel = 2 * np.abs(T23.imag)                       # invariant under rotation

    # ---- deorientation (mode 0 and 1) ----
    teta = 0.5 * np.arctan(2 * T23.real / (T22 - T33 + EPS))
    T11, T12, T13, T22, T23, T33 = unitary_rotation(T11, T12, T13, T22, T23, T33, teta)

    TP = T11 + T22 + T33
    SpanMin = np.full_like(TP, EPS)
    SpanMax = TP

    # ---- volume power under HV_type = 1 (mode 0's default) ----
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio1 = 10 * np.log10((T11 + T22 - 2 * T12.real) / (T11 + T22 + 2 * T12.real + EPS))
    # +-inf / nan compare False, exactly like MATLAB's &&, so no sanitising here
    mid = (ratio1 > -2) & (ratio1 <= 2)
    PPvol = np.where(mid, 2 * (2 * T33 - Phel), (15 / 8) * (2 * T33 - Phel))

    Podd = np.zeros_like(T11)
    Pdbl = np.zeros_like(T11)
    Pvol = PPvol.copy()
    Phelo = Phel.copy()

    # ================= Freeman-Yamaguchi branch (PPvol < 0) =================
    m_neg = PPvol < 0
    if m_neg.any():
        HHHH = (T11[m_neg] + 2 * T12.real[m_neg] + T22[m_neg]) / 2
        HHVV_re = (T11[m_neg] - T22[m_neg]) / 2
        HHVV_im = -T12.imag[m_neg]
        HVHV = T33[m_neg] / 2
        VVVV = (T11[m_neg] - 2 * T12.real[m_neg] + T22[m_neg]) / 2
        H0, V0 = HHHH.copy(), VVVV.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio2 = 10 * np.log10(VVVV / (HHHH + EPS))
        r_lo, r_hi = ratio2 <= -2, ratio2 > 2
        div = np.where(r_lo | r_hi, 15.0, 8.0)
        cH = np.where(r_lo, 8.0, 3.0)
        cV = np.where(r_lo, 3.0, np.where(r_hi, 8.0, 3.0))
        cR = np.where(r_lo | r_hi, 2.0, 1.0)
        fv = np.where(r_lo | r_hi, 15 * HVHV / 4, 8 * HVHV / 2)
        HHHH = HHHH - cH * fv / div
        VVVV = VVVV - cV * fv / div
        HHVV_re = HHVV_re - cR * fv / div
        bad = (HHHH <= EPS) | (VVVV <= EPS)
        # in the bad case the MATLAB restores the subtracted parts, i.e. fv is
        # simply the summed power of the three original channels
        fv = np.where(bad, H0 + HVHV + V0, fv)
        good = ~bad
        H, V, R, I = HHHH[good], VVVV[good], HHVV_re[good], HHVV_im[good]
        rt = R * R + I * I
        over = rt > H * V
        sc = np.sqrt((H * V) / (rt + EPS))
        R = np.where(over, R * sc, R)
        I = np.where(over, I * sc, I)
        pos = R >= 0
        cc = H * V - R * R - I * I
        fd = np.where(pos, cc / (H + V + 2 * R + EPS), 0.0)
        fs = np.where(pos, V - fd, 0.0)
        fd_ng = np.where(~pos, V - cc / (H + V - 2 * R + EPS), 0.0)
        fs_ng = np.where(~pos, cc / (H + V - 2 * R + EPS), 0.0)
        fs = np.where(pos, fs, fs_ng)
        fd = np.where(pos, fd, fd_ng)
        b_re = np.where(pos, (fd + R) / (fs + EPS), 1.0)
        b_im = np.where(pos, I / (fs + EPS), 0.0)
        a_re = np.where(pos, -1.0, (R - fs_ng) / (fd_ng + EPS))
        a_im = np.where(pos, 0.0, I / (fd_ng + EPS))
        sf, df = np.zeros_like(HHHH), np.zeros_like(HHHH)
        sre, sim = np.zeros_like(HHHH), np.zeros_like(HHHH)
        are, aim = np.zeros_like(HHHH), np.zeros_like(HHHH)
        sf[good], df[good] = fs, fd
        sre[good], sim[good] = b_re, b_im
        are[good], aim[good] = a_re, a_im
        sv = fv.copy()
        Ps = sf * (1 + sre ** 2 + sim ** 2)
        Pd = df * (1 + are ** 2 + aim ** 2)
        Pv = sv
        Ps = np.clip(Ps, SpanMin[m_neg], SpanMax[m_neg])
        Pd = np.clip(Pd, SpanMin[m_neg], SpanMax[m_neg])
        Pv = np.clip(Pv, SpanMin[m_neg], SpanMax[m_neg])
        Podd[m_neg], Pdbl[m_neg], Pvol[m_neg] = Ps, Pd, Pv
        Phelo[m_neg] = 0.0                            # Pc = 0 in this branch

    # ================= Yamaguchi four-component branch =====================
    m_pos = ~m_neg
    if m_pos.any():
        T11p, T12p, T13p = T11[m_pos], T12[m_pos], T13[m_pos]
        TPp, Pvolp, Phelp = TP[m_pos], PPvol[m_pos], Phel[m_pos]
        r1 = ratio1[m_pos]
        S = T11p - Pvolp / 2
        D = TPp - Pvolp - Phelp - S
        Cre = T12p.real + T13p.real
        Cim = T12p.imag + T13p.imag
        Cre = np.where(r1 <= -2, Cre - Pvolp / 6, np.where(r1 > 2, Cre + Pvolp / 6, Cre))
        cc = Cre * Cre + Cim * Cim
        CO = 2 * T11p + Phelp - TPp
        pos_co = CO > 0
        yPs0 = np.where(pos_co, S + cc / (S + EPS), S - cc / (D + EPS))
        yPd0 = np.where(pos_co, D - cc / (S + EPS), D + cc / (D + EPS))
        over = (Pvolp + Phelp) > TPp
        neg_s, neg_d = yPs0 < 0, yPd0 < 0
        both = neg_s & neg_d
        rest = TPp - Pvolp - Phelp
        yPs = np.where(neg_s, 0.0, np.where(neg_d, rest, yPs0))
        yPd = np.where(neg_d, 0.0, np.where(neg_s, rest, yPd0))
        Pvol_p = Pvolp.copy()
        yPs = np.where(both, 0.0, yPs)
        yPd = np.where(both, 0.0, yPd)
        Pvol_p = np.where(both, TPp - Phelp, Pvol_p)
        yPs = np.where(over, 0.0, yPs)
        yPd = np.where(over, 0.0, yPd)
        Pvol_p = np.where(over, TPp - Phelp, Pvol_p)
        yPs = np.clip(yPs, EPS, TPp)
        yPd = np.clip(yPd, EPS, TPp)
        Pvol_p = np.clip(Pvol_p, EPS, TPp)
        Podd[m_pos], Pdbl[m_pos], Pvol[m_pos] = yPs, yPd, Pvol_p

    return Podd, Pdbl, Pvol, Phelo
