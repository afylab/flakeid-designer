"""Thin-film optical contrast model for 2D flakes on SiO2/Si.

The reflected intensity of the stack  air | flake (d) | SiO2 (t_ox) | Si
is computed with the standard transfer-matrix method, averaged over the
illumination cone of the objective (s and p polarisation), weighted by the
lamp spectrum and the camera's RGB spectral responses.  The optical
contrast per colour channel is

    C_c(d) = I_c(flake, d) / I_c(substrate) - 1

which is independent of per-channel white balance gains.  A tone curve
(gamma) can be applied to mimic how the camera/JPEG encodes intensities.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Material refractive indices  n + i k   (wavelength in nm)
# ---------------------------------------------------------------------------

# Crystalline Si, Aspnes & Studna (1983)
_SI_TABLE = np.array([
    # lambda,  n,     k
    [380.0, 6.00, 0.90],
    [400.0, 5.570, 0.387],
    [413.3, 5.222, 0.269],
    [427.5, 4.961, 0.203],
    [442.8, 4.753, 0.163],
    [459.2, 4.583, 0.130],
    [476.9, 4.442, 0.109],
    [495.9, 4.320, 0.073],
    [516.6, 4.215, 0.061],
    [539.1, 4.123, 0.048],
    [563.6, 4.042, 0.032],
    [590.4, 3.969, 0.030],
    [619.9, 3.906, 0.022],
    [652.6, 3.847, 0.016],
    [688.8, 3.796, 0.013],
    [729.3, 3.752, 0.010],
    [780.0, 3.705, 0.008],
])

# Graphite, in-plane (ordinary) optical constants, approximate values
# after Djurisic & Li (1999) / Weber et al. (2010).  Graphene layers are
# modelled as graphite with 0.335 nm per layer.
_GRAPHITE_TABLE = np.array([
    [380.0, 2.36, 1.20],
    [400.0, 2.43, 1.23],
    [450.0, 2.56, 1.29],
    [500.0, 2.64, 1.33],
    [550.0, 2.70, 1.37],
    [600.0, 2.75, 1.42],
    [650.0, 2.80, 1.48],
    [700.0, 2.85, 1.54],
    [780.0, 2.92, 1.63],
])


def n_si(wl):
    wl = np.asarray(wl, float)
    return (np.interp(wl, _SI_TABLE[:, 0], _SI_TABLE[:, 1])
            + 1j * np.interp(wl, _SI_TABLE[:, 0], _SI_TABLE[:, 2]))


def n_sio2(wl):
    """Fused silica, Malitson (1965) Sellmeier."""
    l2 = (np.asarray(wl, float) / 1000.0) ** 2
    n2 = (1 + 0.6961663 * l2 / (l2 - 0.0684043 ** 2)
          + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
          + 0.8974794 * l2 / (l2 - 9.896161 ** 2))
    return np.sqrt(n2) + 0j


def n_graphite(wl):
    wl = np.asarray(wl, float)
    return (np.interp(wl, _GRAPHITE_TABLE[:, 0], _GRAPHITE_TABLE[:, 1])
            + 1j * np.interp(wl, _GRAPHITE_TABLE[:, 0], _GRAPHITE_TABLE[:, 2]))


def n_hbn(wl):
    """hBN in-plane index, Cauchy fit (~2.18 at 550 nm, transparent)."""
    lum = np.asarray(wl, float) / 1000.0
    return 2.13 + 0.0149 / lum ** 2 + 0j


def n_mos2(wl):
    """Very rough MoS2 index (placeholder for future materials)."""
    wl = np.asarray(wl, float)
    return 4.8 + 0j + 1j * np.interp(wl, [400, 500, 600, 700], [2.0, 1.3, 1.0, 0.8])


MATERIALS = {
    "graphite": dict(index=n_graphite, layer_nm=0.335, aliases=("graphene", "c")),
    "hbn": dict(index=n_hbn, layer_nm=0.333, aliases=("bn", "h-bn")),
    "mos2": dict(index=n_mos2, layer_nm=0.65, aliases=()),
}


def resolve_material(name: str) -> str:
    key = name.strip().lower().replace("_", "").replace(" ", "")
    for m, info in MATERIALS.items():
        if key == m or key in info["aliases"]:
            return m
    raise ValueError(f"Unknown material '{name}'. Known: {', '.join(MATERIALS)}")


# ---------------------------------------------------------------------------
# Illumination / camera
# ---------------------------------------------------------------------------

def blackbody(wl_nm, temp_k):
    wl = np.asarray(wl_nm, float) * 1e-9
    h, c, kb = 6.626e-34, 2.998e8, 1.381e-23
    return 1.0 / (wl ** 5 * (np.exp(h * c / (wl * kb * temp_k)) - 1.0))


def gaussian(wl, center, fwhm):
    s = fwhm / 2.3548
    return np.exp(-0.5 * ((wl - center) / s) ** 2)


@dataclass
class CameraModel:
    """Spectral model of illumination x camera RGB response."""
    lamp_temp_k: float = 3100.0                 # halogen lamp
    centers_nm: tuple = (600.0, 535.0, 465.0)   # R, G, B peak response
    fwhm_nm: tuple = (80.0, 85.0, 70.0)
    ir_cut_nm: float = 680.0                    # IR-cut filter edge
    gamma: float = 1.0                          # encoded = linear ** (1/gamma)
    ccm: tuple | None = None                    # 3x3 colour-correction matrix on linear RGB
    flare: tuple = (0.0, 0.0, 0.0)              # stray light per channel, in reflectance units

    def weights(self, wl):
        lamp = blackbody(wl, self.lamp_temp_k)
        ircut = 1.0 / (1.0 + np.exp((wl - self.ir_cut_nm) / 8.0))
        w = np.stack([gaussian(wl, c, f) for c, f in zip(self.centers_nm, self.fwhm_nm)])
        w = w * lamp * ircut
        return w / w.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Transfer matrix reflectance
# ---------------------------------------------------------------------------

def _reflectance(n_layers, d_layers, n_sub, wl, cos0, n0=1.0):
    """Reflectance of a multilayer for arrays of wavelength and incidence.

    n_layers: list of complex index arrays (shape (L,)) top to bottom
    d_layers: list of thicknesses (nm), same length (scalar or array (D,))
    Returns array broadcast over (D, A, L) for s and p averaged.
    """
    # The characteristic-matrix form below uses the  N = n - ik  convention.
    n_layers = [np.conj(n) for n in n_layers]
    n_sub = np.conj(n_sub)
    sin0 = np.sqrt(1 - cos0 ** 2)
    kx = n0 * sin0                                   # (A,1)
    out = 0.0
    for pol in ("s", "p"):
        def admittance(n, cos):
            return n * cos if pol == "s" else n / cos

        cos_sub = np.sqrt(1 - (kx / n_sub) ** 2 + 0j)
        eta_sub = admittance(n_sub, cos_sub)
        # characteristic matrix product  M = prod [[cos d, i sin d / eta],[i eta sin d, cos d]]
        B = np.ones(np.broadcast(d_layers[0], kx, wl).shape, complex)
        C = eta_sub * B
        # go from substrate upward:  [B;C] = M_1 ... M_N [1; eta_sub]
        for n, d in reversed(list(zip(n_layers, d_layers))):
            cos = np.sqrt(1 - (kx / n) ** 2 + 0j)
            eta = admittance(n, cos)
            delta = 2 * np.pi * n * d * cos / wl
            cd, sd = np.cos(delta), np.sin(delta)
            B, C = cd * B + 1j * sd / eta * C, 1j * eta * sd * B + cd * C
        eta0 = admittance(n0, cos0)
        r = (eta0 * B - C) / (eta0 * B + C)
        out = out + 0.5 * np.abs(r) ** 2
    return out


@dataclass
class ContrastModel:
    material: str
    oxide_nm: float
    na: float = 0.30
    camera: CameraModel = field(default_factory=CameraModel)
    wl: np.ndarray = field(default_factory=lambda: np.arange(400.0, 721.0, 5.0))
    n_angles: int = 6
    index_table: np.ndarray | None = None   # optional (L,3) [wl, n, k] override for the flake

    def __post_init__(self):
        self.material = resolve_material(self.material)
        self._w = self.camera.weights(self.wl)                 # (3, L)
        # angular sampling of the illumination cone (weights ~ sin(theta))
        th_max = np.arcsin(min(self.na, 0.95))
        th = (np.arange(self.n_angles) + 0.5) / self.n_angles * th_max
        self._cos = np.cos(th)[:, None]                        # (A,1)
        aw = np.sin(th) * np.cos(th)
        self._aw = (aw / aw.sum())[:, None]
        self._n_ox = n_sio2(self.wl)
        self._n_si = n_si(self.wl)
        if self.index_table is not None:
            t = np.asarray(self.index_table, float)
            self._n_fl = np.interp(self.wl, t[:, 0], t[:, 1]) + 1j * np.interp(self.wl, t[:, 0], t[:, 2])
        else:
            self._n_fl = MATERIALS[self.material]["index"](self.wl)
        self._i_sub = self._channel_intensity(None)

    def _channel_intensity(self, d):
        if d is None:
            R = _reflectance([self._n_ox], [self.oxide_nm], self._n_si, self.wl, self._cos)
            R = R[None]
        else:
            d = np.atleast_1d(np.asarray(d, float))[:, None, None]
            R = _reflectance([self._n_fl, self._n_ox], [d, self.oxide_nm],
                             self._n_si, self.wl, self._cos)
        Ra = (R * self._aw[None]).sum(axis=1)                  # (D, L)
        rgb = Ra @ self._w.T                                   # (D, 3)
        if self.camera.ccm is not None:
            rgb = np.maximum(rgb @ np.asarray(self.camera.ccm, float).T, 1e-6)
        # stray light (flare, internal reflections) adds a flake-independent
        # background that reduces all contrasts
        return rgb + np.asarray(self.camera.flare, float)[None, :]

    def substrate_reflectance(self):
        return self._i_sub[0]

    def contrast(self, thickness_nm):
        """RGB contrast (D,3) for an array of flake thicknesses (nm)."""
        ratio = self._channel_intensity(thickness_nm) / self._i_sub
        return ratio ** (1.0 / self.camera.gamma) - 1.0

    @property
    def layer_nm(self):
        return MATERIALS[self.material]["layer_nm"]
