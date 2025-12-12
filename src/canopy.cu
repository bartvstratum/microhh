/*
 * MicroHH
 * Copyright (c) 2011-2020 Chiel van Heerwaarden
 * Copyright (c) 2011-2020 Thijs Heus
 * Copyright (c) 2014-2020 Bart van Stratum
 *
 * This file is part of MicroHH
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "canopy.h"
#include "grid.h"
#include "fields.h"
#include "fast_math.h"
#include "tools.h"


namespace
{
    namespace fm = Fast_math;

    template<typename TF> __global__
    void canopy_drag_1D_kernel(
            TF* const restrict ut,
            TF* const restrict vt,
            TF* const restrict wt,
            const TF* const restrict u,
            const TF* const restrict v,
            const TF* const restrict w,
            const TF* const restrict pad,
            const TF* const restrict padh,
            const TF utrans,
            const TF vtrans,
            const TF cd,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jstride, const int kstride)
    {
        const int ii = 1;
        const int jj = jstride;
        const int kk = kstride;

        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;

                    // Interpolate `v` and `w` to `u` locations.
                    const TF u_on_u = u[ijk] + utrans;
                    const TF v_on_u = TF(0.25) * (v[ijk] + v[ijk-ii] + v[ijk-ii+jj] + v[ijk+jj]) + vtrans;
                    const TF w_on_u = TF(0.25) * (w[ijk] + w[ijk-ii] + w[ijk-ii+kk] + w[ijk+kk]);

                    // Interpolate `u` and `w` to `v` locations.
                    const TF u_on_v = TF(0.25) * (u[ijk] + u[ijk+ii] + u[ijk+ii-jj] + v[ijk-jj]) + utrans;
                    const TF v_on_v = v[ijk] + vtrans;
                    const TF w_on_v = TF(0.25) * (w[ijk] + w[ijk+kk] + w[ijk+kk-kk] + w[ijk-jj]);

                    // Interpolate `u` and `v` to `w` locations.
                    const TF u_on_w = TF(0.25) * (u[ijk] + u[ijk+ii] + u[ijk+ii-kk] + v[ijk-kk]) + utrans;
                    const TF v_on_w = TF(0.25) * (v[ijk] + v[ijk+jj] + v[ijk+jj-kk] + v[ijk-kk]) + vtrans;
                    const TF w_on_w = w[ijk];

                    const TF ftau_u = -cd * pad[k] *
                        std::pow( fm::pow2(u_on_u) +
                                  fm::pow2(v_on_u) +
                                  fm::pow2(w_on_u), TF(0.5) );

                    const TF ftau_v = -cd * pad[k] *
                        std::pow( fm::pow2(u_on_v) +
                                  fm::pow2(v_on_v) +
                                  fm::pow2(w_on_v), TF(0.5) );

                    const TF ftau_w = -cd * padh[k] *
                        std::pow( fm::pow2(u_on_w) +
                                  fm::pow2(v_on_w) +
                                  fm::pow2(w_on_w), TF(0.5) );

                    ut[ijk] += ftau_u * u_on_u;
                    vt[ijk] += ftau_v * v_on_v;
                    wt[ijk] += ftau_w * w_on_w;
                }
    }
}


#ifdef USECUDA
template <typename TF>
void Canopy<TF>::prepare_device()
{
    if (!sw_canopy)
        return;

    auto& gd = grid.get_grid_data();

    if (sw_3d_pad)
    {
        throw std::runtime_error("3D canopy not yet supported on GPU.");
    }
    else
    {
        pad_g.allocate(gd.kcells);
        padh_g.allocate(gd.kcells);

        cuda_safe_call(cudaMemcpy(pad_g, pad.data(), pad_g.size()*sizeof(TF), cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(padh_g, padh.data(), padh_g.size()*sizeof(TF), cudaMemcpyHostToDevice));
    }
}


template <typename TF>
void Canopy<TF>::clear_device()
{
    if (!sw_canopy)
        return;
    
    // No need for cleanup with `cuda_vector`'s.
}


template <typename TF>
void Canopy<TF>::exec()
{
    if (!sw_canopy)
        return;

    auto& gd = grid.get_grid_data();

    const int blocki = gd.ithread_block;
    const int blockj = gd.jthread_block;
    const int gridi  = gd.imax/blocki + (gd.imax%blocki > 0);
    const int gridj  = gd.jmax/blockj + (gd.jmax%blockj > 0);

    dim3 gridGPU (gridi, gridj, kend_canopy);
    dim3 blockGPU(blocki, blockj, 1);

    canopy_drag_1D_kernel<TF><<<gridGPU, blockGPU>>>(
            fields.mt.at("u")->fld_g,
            fields.mt.at("v")->fld_g,
            fields.mt.at("w")->fld_g,
            fields.mp.at("u")->fld_g,
            fields.mp.at("v")->fld_g,
            fields.mp.at("w")->fld_g,
            pad_g,
            padh_g,
            gd.utrans,
            gd.vtrans,
            cd,
            gd.istart, gd.iend,
            gd.jstart, gd.jend,
            gd.kstart, kend_canopy,
            gd.jstride, gd.kstride);
}
#endif


#ifdef FLOAT_SINGLE
template class Canopy<float>;
#else
template class Canopy<double>;
#endif