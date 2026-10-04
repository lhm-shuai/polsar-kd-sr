# -*- coding: utf-8 -*-
"""Per-tile IPSD scan: AUC and inside/outside contrast of SPAN vs the descriptor,
used to choose the tiles the complementarity figure shows (each column must be a
tile where the winning statistic wins on both metrics, so the picture agrees
with the numbers)."""
import glob
import io
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
from scipy import ndimage
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'gapfill'))
from yamaguchi import c3_to_t3, yamaguchi4_mode0          # noqa: E402

IPSD = r'D:\雷达\数据库\IPSD'
OUT = os.path.join(HERE, 'analysis', 'out', 'ipsd_tiles.json')
REF_PSDD = 1.24e-3
C = 0.21
GUARD = 8


def decompose(tif):
    a = tifffile.imread(tif).astype(np.float64)
    T11, T12, T13, T22, T23, T33 = c3_to_t3(
        a[..., 0], a[..., 1] + 1j * a[..., 2], a[..., 3] + 1j * a[..., 4],
        a[..., 5], a[..., 6] + 1j * a[..., 7], a[..., 8])
    Podd, Pdbl, Pvol, Phel = yamaguchi4_mode0(T11, T12, T13, T22, T23, T33)
    return (T11 + T22 + T33), Pdbl + Phel + Pvol


def contrast(img, mask, guard):
    inside = img[mask]
    outside = img[~guard]
    return float(np.median(inside) / max(np.median(outside), 1e-30))


def main():
    rows = []
    xmls = sorted(glob.glob(os.path.join(IPSD, 'VOC', 'Annotations', '*.xml')))
    for i, xml in enumerate(xmls):
        base = os.path.basename(xml)[:-4]
        tif = os.path.join(IPSD, 'VOC', 'TIFImages', base + '.tif')
        if not os.path.exists(tif):
            continue
        r = ET.parse(xml).getroot()
        boxes = []
        for o in r.findall('.//object'):
            b = o.find('bndbox')
            if b is not None:
                boxes.append([float(b.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')])
        if not boxes:
            continue
        span, S = decompose(tif)
        mask = np.zeros(span.shape, bool)
        for x1, y1, x2, y2 in boxes:
            mask[int(max(0, y1)):int(y2) + 1, int(max(0, x1)):int(x2) + 1] = True
        if mask.sum() < 16:
            continue
        guard = ndimage.binary_dilation(mask, iterations=GUARD)
        if (~guard).sum() < 16:
            continue
        k = REF_PSDD / max(np.median(S.ravel()), 1e-30)
        pmv = np.exp(-C / np.maximum(k * S, 1e-30))
        y = np.concatenate([np.ones(mask.sum()), np.zeros((~guard).sum())])
        auc_s = roc_auc_score(y, np.concatenate([span[mask], span[~guard]]))
        auc_p = roc_auc_score(y, np.concatenate([pmv[mask], pmv[~guard]]))
        rows.append(dict(tile=base, boxt=mask.mean(),
                         auc_span=auc_s, auc_pmv=auc_p,
                         ctr_span=contrast(span, mask, guard),
                         ctr_pmv=contrast(pmv, mask, guard)))
        if (i + 1) % 100 == 0:
            print('  %d/%d' % (i + 1, len(xmls)), flush=True)

    with io.open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(rows, fh, indent=1)
    print('wrote', OUT, len(rows), 'tiles')

    sp = sorted(rows, key=lambda r: (r['auc_span'] - r['auc_pmv'], r['ctr_span'] / max(r['ctr_pmv'], 1e-9)),
                reverse=True)
    pm = sorted(rows, key=lambda r: (r['auc_pmv'] - r['auc_span'], r['ctr_pmv'] / max(r['ctr_span'], 1e-9)),
                reverse=True)
    print('\ntop SPAN tiles (auc gap, contrast ratio):')
    for r in sp[:8]:
        print('  %s  dAUC %+.3f  SPAN %.3f Pmv %.3f  ctr ratio %.2f  pct %.3f'
              % (r['tile'], r['auc_span'] - r['auc_pmv'], r['auc_span'], r['auc_pmv'],
                 r['ctr_span'] / max(r['ctr_pmv'], 1e-9), r['boxt']))
    print('\ntop descriptor tiles (auc gap, contrast ratio):')
    for r in pm[:8]:
        print('  %s  dAUC %+.3f  Pmv %.3f SPAN %.3f  ctr ratio %.2f  pct %.3f'
              % (r['tile'], r['auc_pmv'] - r['auc_span'], r['auc_pmv'], r['auc_span'],
                 r['ctr_pmv'] / max(r['ctr_span'], 1e-9), r['boxt']))


if __name__ == '__main__':
    main()
