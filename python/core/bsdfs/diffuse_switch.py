# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines a custom bsdf that switches to diffuse transmittance on shape exit."""

from __future__ import annotations

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore

if not mi.variant():
  mi.set_variant('llvm_ad_rgb')


def _eta_from_specular(specular):
  """Calculates the eta from the specular component.

  Args:
    specular: The specular component of the BSDF.

  Returns:
    The eta of the surface.
  """
  return 2.0 * dr.rcp(1.0 - dr.sqrt(0.08 * specular)) - 1.0


def _sample_diffuse(
    si: mi.SurfaceInteraction3f, sample2: mi.Point2f, active: mi.Bool
) -> tuple[mi.BSDFSample3f, mi.Spectrum]:
  cos_theta_i = mi.Frame3f.cos_theta(si.wi)
  bs = mi.BSDFSample3f()

  bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
  bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
  bs.eta = 1.0
  bs.sampled_type = +mi.BSDFFlags.DiffuseReflection
  bs.sampled_component = 0

  valid = active & (bs.pdf > 0) & (cos_theta_i > 0)
  value = dr.select(valid, 1.0, 0.0)

  return bs, mi.Spectrum(value)


def _eval_diffuse(
    si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: mi.Bool
) -> mi.Spectrum:
  cos_theta_i = mi.Frame3f.cos_theta(si.wi)
  cos_theta_o = mi.Frame3f.cos_theta(wo)

  valid = active & (cos_theta_i > 0) & (cos_theta_o > 0)
  value = dr.select(valid, dr.inv_pi * cos_theta_o, 0.0)

  return mi.Spectrum(value)


def _pdf_diffuse(
    si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: mi.Bool
) -> mi.Float:
  cos_theta_i = mi.Frame3f.cos_theta(si.wi)
  cos_theta_o = mi.Frame3f.cos_theta(wo)
  pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)
  valid = active & (cos_theta_i > 0) & (cos_theta_o > 0)

  return dr.select(valid, pdf, 0.0)


class DiffuseSwitch(mi.BSDF):
  """A BSDF that switches to diffuse transmittance on shape exit.

  This BSDF wraps another BSDF and switches to diffuse transmittance on shape
  entry and/or exit.

  There are four different `diffuse_mode` for switching to diffuse
  transmittance:

  - `entry_exit`: Switches to diffuse transmittance on both entry and exit from
    the shape.
  - `entry_forced_exit`: Switches to diffuse transmittance on entry from the
    shape and forced exit from the shape.
  - `exit`: Switches to diffuse transmittance only on exit from the shape.
  - `forced_exit`: Switches to diffuse transmittance only on forced exit from
  the
    shape.

  The default mode is `entry_forced_exit`.

  Example usage:

  ```python

  bsdf = mi.load_dict({
      'type': 'diffuse_switch',
      'sss_bsdf': {'type': 'principled'},
      'diffuse_mode': 'entry_exit',
  })
  ```
  """

  def __init__(self, props):
    mi.BSDF.__init__(self, props)

    self.sss_bsdf = props['sss_bsdf']
    self.diffuse_mode = props.get('diffuse_mode', 'entry_exit')
    self.entry_nested_prob = props.get('entry_nested_prob', 0.5)

    # Set the BSDF flags
    self.m_flags = self.sss_bsdf.m_flags
    self.m_components = self.sss_bsdf.m_components

    if mi.BSDFFlags.DiffuseTransmission not in self.m_components:
      self.m_components.append(mi.BSDFFlags.DiffuseTransmission)

    if not mi.has_flag(self.sss_bsdf.m_flags, mi.BSDFFlags.DiffuseTransmission):
      self.m_flags |= mi.BSDFFlags.DiffuseTransmission

    self.eta = 1.0
    self.query_specular = False
    if self.sss_bsdf.has_attribute('eta'):
      sss_bsdf_params = mi.traverse(self.sss_bsdf)
      if 'eta.value' not in sss_bsdf_params:
        raise ValueError(
            'Only float eta values are supported, used a specular texture'
            ' instead.'
        )
      self.eta = sss_bsdf_params['eta.value']
    elif self.sss_bsdf.has_attribute('specular'):
      self.query_specular = True

  def _diffuse_transmission_fresnel(self, si, active):
    cos_theta_i = mi.Frame3f.cos_theta(si.wi)

    eta = self.eta
    if self.query_specular:
      eta = _eta_from_specular(
          self.sss_bsdf.eval_attribute_1('specular', si, active)
      )

    fresnel = mi.fresnel(cos_theta_i, eta)[0]
    return fresnel

  def _fresnel_moment_1(self, eta):
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    if eta < 1:
      return (
          0.45966
          - 1.73965 * eta
          + 3.37668 * eta2
          - 3.904945 * eta3
          + 2.49277 * eta4
          - 0.68441 * eta5
      )
    else:
      return (
          -4.61686
          + 11.1136 * eta
          - 10.4646 * eta2
          + 5.11455 * eta3
          - 1.27198 * eta4
          + 0.12746 * eta5
      )

  def _sw(self, si, w, active):
    eta = self.eta
    if self.query_specular:
      eta = _eta_from_specular(
          self.sss_bsdf.eval_attribute_1('specular', si, active)
      )

    c = 1 - 2 * self._fresnel_moment_1(1 / eta)
    # return (1 - mi.fresnel(mi.Frame3f.cos_theta(w), eta)[0]) / (c)
    return (1 - mi.fresnel(dr.abs(mi.Frame3f.cos_theta(w)), eta)[0]) / (c)

  def sample_diffuse_entry_exit(self, ctx, si, sample1, sample2, active):
    diffuse_fresnel = self._diffuse_transmission_fresnel(si, active)
    sampled_reflection = active & (sample1 < diffuse_fresnel)
    sampled_refraction = active & (sample1 >= diffuse_fresnel)

    # Reweight sample1 to reuse the sample1 if the nested bsdf needs it
    sample1 = dr.select(
        sampled_reflection,
        dr.clip(sample1 / diffuse_fresnel, 0, 1),
        (diffuse_fresnel - sample1) / (1 - diffuse_fresnel),
    )
    # Sample reflection
    ctx_nested = mi.BSDFContext()
    ctx_nested.type_mask = mi.BSDFFlags.Reflection

    bs_reflection, weight_reflection = self.sss_bsdf.sample(
        ctx_nested, si, sample1, sample2, sampled_reflection
    )

    # Sample refraction
    swap_wi = sampled_refraction & (mi.Frame3f.cos_theta(si.wi) < 0.0)
    swap_wo = sampled_refraction & (mi.Frame3f.cos_theta(si.wi) > 0.0)
    si.wi.z[swap_wi] *= -1
    bs_refraction, weight_refraction = _sample_diffuse(
        si, sample2, sampled_refraction
    )
    if dr.hint(dr.grad_enabled(diffuse_fresnel), mode='scalar'):
      weight_refraction[weight_refraction > 0] = dr.replace_grad(
          weight_refraction,
          (1 - diffuse_fresnel) / (1 - dr.detach(diffuse_fresnel)),
      )
    # We are perfectly importance sampling the diffuse refraction, so the weight
    # scaling for the refraction is just 1, for the reflection we need to
    # account for the probabilit of samplign that lobe which differs from
    # the fresnel of the nested bsdf.
    si.wi.z[swap_wi] *= -1
    bs_refraction.wo.z[swap_wo] *= -1

    bs = dr.select(sampled_reflection, bs_reflection, bs_refraction)
    weight_reflection /= dr.detach(diffuse_fresnel)

    weight = dr.select(sampled_reflection, weight_reflection, weight_refraction)
    # Multiply the nested pdfs with the probability given by the fresnel
    # maximum to avoid floating point errors
    bs.pdf *= dr.detach(
        dr.select(
            sampled_reflection,
            dr.maximum(diffuse_fresnel, 0.0),
            dr.maximum(1.0 - diffuse_fresnel, 0.0),
        )
    )

    return bs, weight

  def eval_diffuse_entry_exit(self, ctx, si, wo, active):
    cos_theta_i = mi.Frame3f.cos_theta(si.wi)
    cos_theta_o = mi.Frame3f.cos_theta(wo)
    eval_diff_trans = cos_theta_i * cos_theta_o < 0.0 & active
    eval_nested = cos_theta_i * cos_theta_o > 0.0 & active
    ctx_nested = mi.BSDFContext()
    ctx_nested.type_mask = mi.BSDFFlags.Reflection
    value = self.sss_bsdf.eval(ctx_nested, si, wo, eval_nested)

    # Fresnel for diffuse transmission
    diffuse_fresnel = self._diffuse_transmission_fresnel(si, eval_diff_trans)

    swap_wi = eval_diff_trans & (cos_theta_i < 0.0)
    swap_wo = eval_diff_trans & (cos_theta_o < 0.0)
    si.wi.z[swap_wi] *= -1
    wo.z[swap_wo] *= -1

    value_diffuse = _eval_diffuse(si, wo, eval_diff_trans)
    # maximum to avoid floating point errors
    value_diffuse[eval_diff_trans] *= dr.maximum(1.0 - diffuse_fresnel, 0.0)
    value[eval_diff_trans] = value_diffuse

    return value

  def pdf_diffuse_entry_exit(self, ctx, si, wo, active):
    diffuse_fresnel = self._diffuse_transmission_fresnel(si, active)

    cos_theta_i = mi.Frame3f.cos_theta(si.wi)
    cos_theta_o = mi.Frame3f.cos_theta(wo)
    eval_pdf_diffuse = cos_theta_i * cos_theta_o < 0.0 & active
    eval_pdf_nested = cos_theta_i * cos_theta_o > 0.0 & active

    # We can only sample reflection of the nested bsdf
    ctx_nested = mi.BSDFContext()
    ctx_nested.type_mask = mi.BSDFFlags.Reflection
    pdf_nested = self.sss_bsdf.pdf(ctx_nested, si, wo, eval_pdf_nested)

    swap_wi = eval_pdf_diffuse & (cos_theta_i < 0.0)
    swap_wo = eval_pdf_diffuse & (cos_theta_o < 0.0)
    si.wi.z[swap_wi] *= -1
    wo.z[swap_wo] *= -1
    pdf_diffuse = _pdf_diffuse(si, wo, eval_pdf_diffuse)

    # Scale with fresnel and solid angle compression
    pdf = dr.select(eval_pdf_nested, pdf_nested, pdf_diffuse)
    # maximum to avoid floating point errors
    pdf *= dr.select(
        eval_pdf_nested,
        dr.maximum(diffuse_fresnel, 0.0),
        dr.maximum(1.0 - diffuse_fresnel, 0.0),
    )

    return pdf

  def sample_diffuse_entry_forced_exit(self, ctx, si, sample1, sample2, active):
    sample_diff_trans = mi.Frame3f.cos_theta(si.wi) < 0.0 & active
    sample_nested = mi.Frame3f.cos_theta(si.wi) > 0.0 & active

    bs, value = self.sss_bsdf.sample(ctx, si, sample1, sample2, sample_nested)
    # We can only sample reflection of the nested bsdf, adjust the pdf to
    # account for this.
    ctx_nested = mi.BSDFContext()
    ctx_nested.type_mask = mi.BSDFFlags.Reflection
    bs.pdf = self.sss_bsdf.pdf(ctx_nested, si, bs.wo, active)

    sampled_transmission = mi.has_flag(
        bs.sampled_type, mi.BSDFFlags.GlossyTransmission
    )
    sample_diff_trans |= sampled_transmission

    swap_wi = sample_diff_trans & (mi.Frame3f.cos_theta(si.wi) < 0.0)
    swap_wo = sampled_transmission & (mi.Frame3f.cos_theta(bs.wo) < 0.0)
    si.wi.z[swap_wi] *= -1
    bs_diffuse, value_diffuse = _sample_diffuse(si, sample2, sample_diff_trans)
    bs_diffuse.wo.z[swap_wo] *= -1
    bs_diffuse.eta = bs.eta

    bs[sample_diff_trans] = bs_diffuse
    value[sample_diff_trans] = value_diffuse

    return bs, value

  def eval_diffuse_entry_forced_exit(self, ctx, si, wo, active):
    eval_diff_trans = (
        mi.Frame3f.cos_theta(si.wi) * mi.Frame3f.cos_theta(wo) < 0.0 & active
    )
    eval_nested = (
        mi.Frame3f.cos_theta(si.wi) * mi.Frame3f.cos_theta(wo) > 0.0 & active
    )

    value = self.sss_bsdf.eval(ctx, si, wo, eval_nested)

    swap_wi = eval_diff_trans & (mi.Frame3f.cos_theta(si.wi) < 0.0)
    swap_wo = eval_diff_trans & (mi.Frame3f.cos_theta(wo) < 0.0)
    si.wi.z[swap_wi] *= -1
    wo.z[swap_wo] *= -1
    value_diffuse = _eval_diffuse(si, wo, eval_diff_trans)
    value[eval_diff_trans] = value_diffuse

    return value

  def pdf_diffuse_entry_forced_exit(self, ctx, si, wo, active):
    pdf_diff_trans = (
        mi.Frame3f.cos_theta(si.wi) * mi.Frame3f.cos_theta(wo) < 0.0 & active
    )
    pdf_nested = (
        mi.Frame3f.cos_theta(si.wi) * mi.Frame3f.cos_theta(wo) > 0.0 & active
    )

    # We can only sample reflection of the nested bsdf
    ctx_nested = mi.BSDFContext()
    ctx_nested.type_mask = mi.BSDFFlags.Reflection
    pdf = self.sss_bsdf.pdf(ctx_nested, si, wo, pdf_nested)

    swap_wi = pdf_diff_trans & (mi.Frame3f.cos_theta(si.wi) < 0.0)
    swap_wo = pdf_diff_trans & (mi.Frame3f.cos_theta(wo) < 0.0)
    si.wi.z[swap_wi] *= -1
    wo.z[swap_wo] *= -1
    pdf_diffuse = _pdf_diffuse(si, wo, pdf_diff_trans)

    pdf[pdf_diff_trans] = pdf_diffuse

    return pdf

  def sample_diffuse_forced_exit(self, ctx, si, sample1, sample2, active):
    sample_diff_trans = mi.Frame3f.cos_theta(si.wi) < 0.0 & active
    sample_nested = mi.Frame3f.cos_theta(si.wi) > 0.0 & active

    bs, value = self.sss_bsdf.sample(ctx, si, sample1, sample2, sample_nested)

    si.wi.z[sample_diff_trans] *= -1
    bs_diffuse, value_diffuse = _sample_diffuse(si, sample2, sample_diff_trans)

    # Set eta on exit
    eta = self.eta
    if self.query_specular:
      eta = _eta_from_specular(
          self.sss_bsdf.eval_attribute_1('specular', si, active)
      )
    bs_diffuse.eta = dr.rcp(eta)

    bs[sample_diff_trans] = bs_diffuse
    # eta^2 accounts for the solid angle compression when crossing the interface
    value[sample_diff_trans] = value_diffuse * dr.square(eta)

    return bs, value

  def eval_diffuse_forced_exit(self, ctx, si, wo, active):
    eval_diff_trans = mi.Frame3f.cos_theta(si.wi) < 0.0 & active
    eval_nested = mi.Frame3f.cos_theta(si.wi) > 0.0 & active

    nested_value = self.sss_bsdf.eval(ctx, si, wo, eval_nested)

    swap_wo = eval_diff_trans & (mi.Frame3f.cos_theta(wo) < 0.0)
    si.wi.z[eval_diff_trans] *= -1
    wo.z[swap_wo] *= -1
    # eta^2 accounts for the solid angle compression when crossing the interface
    eta = self.eta
    if self.query_specular:
      eta = _eta_from_specular(
          self.sss_bsdf.eval_attribute_1('specular', si, active)
      )
    diffuse_value = _eval_diffuse(si, wo, eval_diff_trans) * dr.square(eta)
    value = dr.select(eval_nested, nested_value, diffuse_value)

    return value

  def pdf_diffuse_forced_exit(self, ctx, si, wo, active):
    pdf_diff_trans = mi.Frame3f.cos_theta(si.wi) < 0.0 & active
    pdf_nested = mi.Frame3f.cos_theta(si.wi) > 0.0 & active

    nested_pdf = self.sss_bsdf.pdf(ctx, si, wo, pdf_nested)

    si.wi.z[pdf_diff_trans] *= -1
    pdf_diffuse = _pdf_diffuse(si, wo, pdf_diff_trans)
    pdf = dr.select(pdf_nested, nested_pdf, pdf_diffuse)

    return pdf

  def sample_diffuse_exit(self, ctx, si, sample1, sample2, active):
    bs, value = self.sss_bsdf.sample(ctx, si, sample1, sample2, active)
    sampled_transmission = mi.has_flag(
        bs.sampled_type, mi.BSDFFlags.GlossyTransmission
    )
    sample_diffuse_exit = (
        sampled_transmission & (mi.Frame3f.cos_theta(bs.wo) > 0.0) & active
    )

    si.wi.z[sample_diffuse_exit] *= -1
    bs_diffuse, value_diffuse = _sample_diffuse(
        si, sample2, sample_diffuse_exit
    )
    bs_diffuse.eta = bs.eta

    bs[sample_diffuse_exit] = bs_diffuse
    value[sample_diffuse_exit] = value_diffuse

    return bs, value

  def sample(self, ctx, si_, sample1, sample2, active):
    si = mi.SurfaceInteraction3f(si_)

    if dr.hint(self.diffuse_mode == 'entry_exit', mode='scalar'):
      return self.sample_diffuse_entry_exit(ctx, si, sample1, sample2, active)
    if dr.hint(self.diffuse_mode == 'entry_forced_exit', mode='scalar'):
      return self.sample_diffuse_entry_forced_exit(
          ctx, si, sample1, sample2, active
      )
    if dr.hint(self.diffuse_mode == 'exit', mode='scalar'):
      return self.sample_diffuse_exit(ctx, si, sample1, sample2, active)
    if dr.hint(self.diffuse_mode == 'forced_exit', mode='scalar'):
      return self.sample_diffuse_forced_exit(ctx, si, sample1, sample2, active)

  def eval(self, ctx, si_, wo, active):
    si = mi.SurfaceInteraction3f(si_)

    if dr.hint(self.diffuse_mode == 'entry_exit', mode='scalar'):
      return self.eval_diffuse_entry_exit(ctx, si, wo, active)
    if dr.hint(self.diffuse_mode == 'entry_forced_exit', mode='scalar'):
      return self.eval_diffuse_entry_forced_exit(ctx, si, wo, active)
    if dr.hint(self.diffuse_mode == 'forced_exit', mode='scalar'):
      return self.eval_diffuse_forced_exit(ctx, si, wo, active)

  def pdf(self, ctx, si_, wo, active):
    si = mi.SurfaceInteraction3f(si_)

    if dr.hint(self.diffuse_mode == 'entry_exit', mode='scalar'):
      return self.pdf_diffuse_entry_exit(ctx, si, wo, active)
    if dr.hint(self.diffuse_mode == 'entry_forced_exit', mode='scalar'):
      return self.pdf_diffuse_entry_forced_exit(ctx, si, wo, active)
    if dr.hint(self.diffuse_mode == 'forced_exit', mode='scalar'):
      return self.pdf_diffuse_forced_exit(ctx, si, wo, active)

  def eval_pdf(self, ctx, si_, wo, active):
    si = mi.SurfaceInteraction3f(si_)

    return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

  def has_attribute(self, name, active):
    return self.sss_bsdf.has_attribute(name, active)

  def eval_attribute(self, name, si, active):
    return self.sss_bsdf.eval_attribute(name, si, active)

  def traverse(self, callback):
    # This ensures no extra bsdf identifier name is added in the traversal callback
    self.sss_bsdf.traverse(callback)

  def parameters_changed(self, keys):
    self.sss_bsdf.parameters_changed(keys)

  def to_string(self):
    return 'DiffuseSwitch[\n    sss_bsdf=%s, \n]' % (self.sss_bsdf)


def register():
  """Registers the diffuse exit integrator plugin."""
  mi.register_bsdf(
      'diffuse_switch',
      lambda props: DiffuseSwitch(props),  # pylint: disable=unnecessary-lambda
  )
