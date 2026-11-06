/*
 * MicroHH
 * Copyright (c) 2011-2024 Chiel van Heerwaarden
 * Copyright (c) 2011-2024 Thijs Heus
 * Copyright (c) 2014-2024 Bart van Stratum
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

#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>

#include "master.h"
#include "grid.h"
#include "fields.h"
#include "diff.h"
#include "stats.h"
#include "cross.h"
#include "column.h"
#include "dump.h"
#include "thermo.h"
#include "thermo_moist_functions.h"

#include "constants.h"
#include "microphys.h"
#include "microphys_nsw6.h"
#include "microphys_2mom_warm.h"

// Constants, move out later.
namespace
{
    template<typename TF> constexpr TF qv_min = 1.e-7; // Threshold qv for calculating microphysical terms.
    template<typename TF> constexpr TF ql_min = 1.e-7; // Threshold ql for calculating microphysical terms.
    template<typename TF> constexpr TF qi_min = 1.e-7; // Threshold qi for calculating microphysical terms.
    template<typename TF> constexpr TF qr_min = 1.e-12; // Threshold qr for calculating microphysical terms.
    template<typename TF> constexpr TF qs_min = 1.e-12; // Threshold qs for calculating microphysical terms.
    template<typename TF> constexpr TF qg_min = 1.e-12; // Threshold qg for calculating microphysical terms.
    template<typename TF> constexpr TF q_tiny = 1.e-15; // Threshold qg for calculating microphysical terms.

    template<typename TF> constexpr TF pi = M_PI; // Pi constant.
    template<typename TF> constexpr TF pi_2 = M_PI*M_PI; // Pi constant squared.

    template<typename TF> constexpr TF rho_w = 1.e3; // Density of water.
    template<typename TF> constexpr TF rho_s = 1.e2; // Density of snow.
    template<typename TF> constexpr TF rho_g = 4.e2; // Density of graupel.

    template<typename TF> constexpr TF N_0r = 8.e6; // Intercept parameter rain (m-4).
    template<typename TF> constexpr TF N_0s = 3.e6; // Intercept parameter snow (m-4).
    template<typename TF> constexpr TF N_0g = 4.e6; // Intercept parameter graupel (m-4).

    template<typename TF> constexpr TF a_r = M_PI*rho_w<TF>/6.; // Empirical constant for m_r.
    template<typename TF> constexpr TF a_s = M_PI*rho_s<TF>/6.; // Empirical constant for m_s.
    template<typename TF> constexpr TF a_g = M_PI*rho_g<TF>/6.; // Empirical constant for m_g.

    template<typename TF> constexpr TF b_r = 3.; // Empirical constant for m_r.
    template<typename TF> constexpr TF b_s = 3.; // Empirical constant for m_s.
    template<typename TF> constexpr TF b_g = 3.; // Empirical constant for m_g.

    template<typename TF> constexpr TF c_r = 130.; // Empirical constant for v_r (wrong value in Tomita's paper).
    template<typename TF> constexpr TF c_s = 4.84; // Empirical constant for v_s.
    template<typename TF> constexpr TF c_g = 82.5; // Empirical constant for v_g.

    template<typename TF> constexpr TF d_r = 0.5;  // Empirical constant for v_r.
    template<typename TF> constexpr TF d_s = 0.25; // Empirical constant for v_s.
    template<typename TF> constexpr TF d_g = 0.5;  // Empirical constant for v_g.

    template<typename TF> constexpr TF C_i = 2006.; // Specific heat of solid water.
    template<typename TF> constexpr TF C_l = 4218.; // Specific heat of liquid water.

    template<typename TF> constexpr TF f_1r = 0.78; // First coefficient of ventilation factor for rain.
    template<typename TF> constexpr TF f_1s = 0.65; // First coefficient of ventilation factor for snow.
    template<typename TF> constexpr TF f_1g = 0.78; // First coefficient of ventilation factor for graupel.

    template<typename TF> constexpr TF f_2r = 0.27; // First coefficient of ventilation factor for rain.
    template<typename TF> constexpr TF f_2s = 0.39; // First coefficient of ventilation factor for snow.
    template<typename TF> constexpr TF f_2g = 0.27; // First coefficient of ventilation factor for graupel.

    template<typename TF> constexpr TF E_ri = 1.;  // Collection efficiency of ice for rain.
    template<typename TF> constexpr TF E_rw = 1.;  // Collection efficiency of rain for cloud water.
    template<typename TF> constexpr TF E_sw = 1.;  // Collection efficiency of snow for cloud water.
    template<typename TF> constexpr TF E_gw = 1.;  // Collection efficiency of graupel for cloud water.
    template<typename TF> constexpr TF E_gi = 0.1; // Collection efficiency of graupel for cloud ice.
    template<typename TF> constexpr TF E_sr = 1.;  // Collection efficiency of snow for rain.
    // template<typename TF> constexpr TF E_gr = 1.;  // Collection efficiency of graupel for rain.
    template<typename TF> constexpr TF E_gr = 0.1; // CvH: I reduced the value as the current convection is too active in cold pools, leading to conversion from rain to graupel that extends too far to the surface.

    template<typename TF> constexpr TF K_a = 2.43e-2;  // Thermal diffusion coefficient of air.
    template<typename TF> constexpr TF K_d = 2.26e-5;  // Diffusion coefficient of water vapor in air.

    template<typename TF> constexpr TF M_i = 4.19e-13; // Mass of one cloud ice particle.

    template<typename TF> constexpr TF beta_saut = 6.e-3; // Values from SCALE-LES model.
    template<typename TF> constexpr TF beta_gaut = 0.e-3; // Values from SCALE-LES model.

    template<typename TF> constexpr TF gamma_sacr = 25.e-3;
    template<typename TF> constexpr TF gamma_saut = 60.e-3; // Tomita's code in SCALE is different than paper (0.025).
    template<typename TF> constexpr TF gamma_gacs = 90.e-3;
    template<typename TF> constexpr TF gamma_gaut = 90.e-3;

    template<typename TF> constexpr TF nu = 1.5e-5; // Kinematic viscosity of air.
}

namespace
{
    using namespace Constants;
    using namespace Thermo_moist_functions;
    using namespace Fast_math;
    using Micro_2mom_warm_functions::minmod;

    // Constants of the conversion terms at one height level.
    template<typename TF>
    struct Conversion_constants
    {
        TF lambda_fac_r;
        TF lambda_fac_s;
        TF lambda_fac_g;
        TF V_T_fac_r_k;
        TF V_T_fac_s_k;
        TF V_T_fac_g_k;
        TF fac_iacr;
        TF fac_raci;
        TF fac_racw;
        TF fac_sacw;
        TF fac_saci;
        TF fac_gacw;
        TF fac_gaci;
        TF fac_racs0;
        TF fac_sacr0;
        TF fac_gacr0;
        TF fac_gacs0;
        TF g_s1;
        TF g_s2;
        TF g_s3;
        TF g_r1;
        TF g_r2;
        TF g_r3;
        TF vent_1r;
        TF vent_1s;
        TF vent_1g;
        TF vent_2r;
        TF vent_2s;
        TF vent_2g;
        TF fac_raut;
        TF rho_k;
        TF rho_i;
        TF exner_k;
        TF p_k;
    };

    // Process rates and tendencies of nb compacted cells at one height level.
    template<typename TF>
    void conversion_block(
            TF* const restrict dqr_c, TF* const restrict dqs_c, TF* const restrict dqg_c,
            TF* const restrict dqt_c, TF* const restrict dthl_c, TF* const restrict frz_c,
            const TF* const restrict qr_c, const TF* const restrict qs_c, const TF* const restrict qg_c,
            const TF* const restrict ql_c, const TF* const restrict qi_c,
            const TF* const restrict qt_c, const TF* const restrict thl_c,
            const Conversion_constants<TF> cst, const TF dt, const int nb)
    {
        const TF lambda_fac_r = cst.lambda_fac_r;
        const TF lambda_fac_s = cst.lambda_fac_s;
        const TF lambda_fac_g = cst.lambda_fac_g;
        const TF V_T_fac_r_k = cst.V_T_fac_r_k;
        const TF V_T_fac_s_k = cst.V_T_fac_s_k;
        const TF V_T_fac_g_k = cst.V_T_fac_g_k;
        const TF fac_iacr = cst.fac_iacr;
        const TF fac_raci = cst.fac_raci;
        const TF fac_racw = cst.fac_racw;
        const TF fac_sacw = cst.fac_sacw;
        const TF fac_saci = cst.fac_saci;
        const TF fac_gacw = cst.fac_gacw;
        const TF fac_gaci = cst.fac_gaci;
        const TF fac_racs0 = cst.fac_racs0;
        const TF fac_sacr0 = cst.fac_sacr0;
        const TF fac_gacr0 = cst.fac_gacr0;
        const TF fac_gacs0 = cst.fac_gacs0;
        const TF g_s1 = cst.g_s1;
        const TF g_s2 = cst.g_s2;
        const TF g_s3 = cst.g_s3;
        const TF g_r1 = cst.g_r1;
        const TF g_r2 = cst.g_r2;
        const TF g_r3 = cst.g_r3;
        const TF vent_1r = cst.vent_1r;
        const TF vent_1s = cst.vent_1s;
        const TF vent_1g = cst.vent_1g;
        const TF vent_2r = cst.vent_2r;
        const TF vent_2s = cst.vent_2s;
        const TF vent_2g = cst.vent_2g;
        const TF fac_raut = cst.fac_raut;
        const TF rho_k = cst.rho_k;
        const TF rho_i = cst.rho_i;
        const TF exner_k = cst.exner_k;
        const TF p_k = cst.p_k;

        // Process rates and the state after one microphysics time step.
        #pragma GCC ivdep
        for (int n=0; n<nb; ++n)
        {
            // T from thlE with known ql and qi: positive root of T^2 - tl*T - tl*A = 0, A = (Lv*ql + Ls*qi)/cp.
            const TF tl = exner_k*thl_c[n];
            const TF A  = (Lv<TF>*ql_c[n] + Ls<TF>*qi_c[n]) / cp<TF>;
            const TF T  = TF(0.5) * (tl + std::sqrt(tl*tl + TF(4.)*tl*A));
            const TF qv = qt_c[n] - ql_c[n] - qi_c[n];

            // Flag the sign of the absolute temperature.
            const TF T_pos = TF(T >= T0<TF>);
            const TF T_neg = TF(1.) - T_pos;

            // Check which species are present.
            const bool has_vapor   = (qv      > qv_min<TF>);
            const bool has_liq     = (ql_c[n] > ql_min<TF>);
            const bool has_ice     = (qi_c[n] > qi_min<TF>);
            const bool has_rain    = (qr_c[n] > qr_min<TF>);
            const bool has_snow    = (qs_c[n] > qs_min<TF>);
            const bool has_graupel = (qg_c[n] > qg_min<TF>);

            // Tomita Eq. 27. Use 1/lambda: powers of lambda overflow in single precision for small q; powers of 1/lambda underflow to 0.
            const TF lambda_r_i = std::sqrt(std::sqrt(rho_k * (qr_c[n] + q_tiny<TF>) * lambda_fac_r));
            const TF lambda_s_i = std::sqrt(std::sqrt(rho_k * (qs_c[n] + q_tiny<TF>) * lambda_fac_s));
            const TF lambda_g_i = std::sqrt(std::sqrt(rho_k * (qg_c[n] + q_tiny<TF>) * lambda_fac_g));

            // Fractional powers of 1/lambda.
            const TF lr_12 = std::sqrt(lambda_r_i);
            const TF lr_14 = std::sqrt(lr_12);
            const TF ls_12 = std::sqrt(lambda_s_i);
            const TF ls_14 = std::sqrt(ls_12);
            const TF ls_18 = std::sqrt(ls_14);
            const TF lg_12 = std::sqrt(lambda_g_i);
            const TF lg_14 = std::sqrt(lg_12);

            const TF lr_2 = pow2(lambda_r_i);
            const TF lr_3 = lr_2 * lambda_r_i;
            const TF lr_4 = pow2(lr_2);
            const TF lr_5 = lr_4 * lambda_r_i;
            const TF lr_6 = lr_3 * lr_3;
            const TF ls_2 = pow2(lambda_s_i);
            const TF ls_3 = ls_2 * lambda_s_i;
            const TF ls_4 = pow2(ls_2);
            const TF ls_5 = ls_4 * lambda_s_i;
            const TF ls_6 = ls_3 * ls_3;
            const TF lg_2 = pow2(lambda_g_i);
            const TF lg_3 = lg_2 * lambda_g_i;

            // Tomita Eq. 28
            const TF V_Tr = !(has_rain)    ? TF(0.) : V_T_fac_r_k * lr_12;
            const TF V_Ts = !(has_snow)    ? TF(0.) : V_T_fac_s_k * ls_14;
            const TF V_Tg = !(has_graupel) ? TF(0.) : V_T_fac_g_k * lg_12;

            // ACCRETION
            // Tomita Eq. 29
            const TF P_iacr = !(has_rain && has_ice) ? TF(0.) :
                fac_iacr * lr_6 * lr_12 * qi_c[n];

            // Tomita Eq. 30
            const TF delta_1 = TF(qr_c[n] >= TF(1.e-4));

            // Tomita Eq. 31
            TF P_iacr_s = (TF(1.) - delta_1) * P_iacr;
            TF P_iacr_g = delta_1 * P_iacr;

            // Tomita Eq. 32
            const TF P_raci = !(has_rain && has_ice) ? TF(0.) :
                fac_raci * lr_3 * lr_12 * qi_c[n];

            // Tomita Eq. 33
            TF P_raci_s = (TF(1.) - delta_1) * P_raci;
            TF P_raci_g = delta_1 * P_raci;

            // Tomita Eq. 34, 35
            TF P_racw = !(has_liq && has_rain) ? TF(0.) :
                fac_racw * lr_3 * lr_12 * ql_c[n];
            TF P_sacw = !(has_liq && has_snow) ? TF(0.) :
                fac_sacw * ls_3 * ls_14 * ql_c[n];

            // Tomita Eq. 39
            const TF E_si = std::exp(gamma_sacr<TF> * (T - T0<TF>));

            // Tomita Eq. 36 - 38
            TF P_saci = !(has_snow && has_ice) ? TF(0.) :
                fac_saci * E_si * ls_3 * ls_14 * qi_c[n];
            TF P_gacw = !(has_graupel && has_liq) ? TF(0.) :
                fac_gacw * lg_3 * lg_12 * ql_c[n];
            TF P_gaci = !(has_graupel && has_ice) ? TF(0.) :
                fac_gaci * lg_3 * lg_12 * qi_c[n];

            // Accretion of falling hydrometeors.
            // Tomita Eq. 42
            const TF delta_2 = ( (qr_c[n] >= TF(1.e-4)) || (qs_c[n] >= TF(1.e-4)) ) ? TF(0.) : TF(1.);

            // Tomita Eq. 41
            TF P_racs = !(has_rain && has_snow) ? TF(0.) :
                (TF(1.) - delta_2)
                * fac_racs0 * std::abs(V_Tr - V_Ts) * rho_i
                * ( g_s3 * ls_6 * lambda_r_i
                  + g_s2 * ls_5 * lr_2
                  + g_s1 * ls_4 * lr_3 );

            // Tomita Eq. 44
            const TF P_sacr = !(has_snow && has_rain) ? TF(0.) :
                  fac_sacr0 * std::abs(V_Ts - V_Tr) * rho_i
                * ( g_r1 * lr_4 * ls_3
                  + g_r2 * lr_5 * ls_2
                  + g_r3 * lr_6 * lambda_s_i );

            // Tomita Eq. 43
            TF P_sacr_g = (TF(1.) - delta_2) * P_sacr;
            TF P_sacr_s = delta_2 * P_sacr;

            // Tomita Eq. 49
            const TF E_gs = std::min( TF(1.), std::exp(gamma_gacs<TF> * (T - T0<TF>)) );

            // Tomita Eq. 47
            TF P_gacr = !(has_graupel && has_rain) ? TF(0.) :
                  fac_gacr0 * std::abs(V_Tg - V_Tr) * rho_i
                * ( g_r1 * lr_4 * lg_3
                  + g_r2 * lr_5 * lg_2
                  + g_r3 * lr_6 * lambda_g_i );

            // Tomita Eq. 48
            TF P_gacs = !(has_graupel && has_snow) ? TF(0.) :
                  fac_gacs0 * std::abs(V_Tg - V_Ts) * E_gs * rho_i
                * ( g_s1 * ls_4 * lg_3
                  + g_s2 * ls_5 * lg_2
                  + g_s3 * ls_6 * lambda_g_i );

            // AUTOCONVERSION.
            constexpr TF q_icrt = TF(0.);
            constexpr TF q_scrt = TF(6.e-4);

            // Tomita Eq. 53
            const TF beta_1 = std::min( beta_saut<TF>, beta_saut<TF>*std::exp(gamma_saut<TF> * (T - T0<TF>)) );

            // Tomita Eq. 54
            const TF beta_2 = std::min( beta_gaut<TF>, beta_gaut<TF>*std::exp(gamma_gaut<TF> * (T - T0<TF>)) );

//                    // Tomita Eq. 50. Our Nc0 is SI units, so conversion is applied.
//                    TF P_raut = !(has_liq) ? TF(0.) :
//                        TF(16.7)/rho_k * pow2(rho_k*ql_c[n]) / (TF(5.) + TF(3.66e-2) * TF(1.e-6)*Nc0 / (D_d*rho_k*ql_c[n]));

             // Kharoutdinov and Kogan autoconversion.
             TF P_raut = (has_liq) ?
                 fac_raut * std::pow(ql_c[n], TF(2.47))
                 : TF(0.);

            // Seifert and Beheng autoconversion.
            // const TF x_star = TF(2.6e-10); // SB06, list of symbols, same as UCLA-LES
            // const TF k_cc = TF(9.44e9); // UCLA-LES (Long, 1974), 4.44e9 in SB06, p48
            // const TF nu_c = TF(1.); // SB06, Table 1., same as UCLA-LES
            // const TF kccxs = k_cc / (TF(20.) * x_star) * (nu_c+2)*(nu_c+4) / pow2(nu_c+1);
            // const TF xc  = rho_k * ql_c[n] / Nc0; // Mean mass of cloud drops [kg]
            // const TF tau = TF(1.) - ql_c[n] / (ql_c[n] + qr_c[n] + dsmall); // SB06, Eq 5
            // const TF phi_au = TF(600.) * std::pow(tau, TF(0.68)) * pow3(TF(1.) - pow(tau, TF(0.68))); // UCLA-LES

            // TF P_raut = rho_k * kccxs * pow(ql_c[n], 2) * pow(xc, 2) * (TF(1.) + phi_au / pow2(TF(1.)-tau)); // SB06, eq 4

            // Tomita Eq. 52
            TF P_saut = !(has_ice) ? TF(0.) :
                std::max(beta_1*(qi_c[n] - q_icrt), TF(0.));

            // Tomita Eq. 54
            TF P_gaut = !(has_snow) ? TF(0.) :
                std::max(beta_2*(qs_c[n] - q_scrt), TF(0.));

            // PHASE CHANGES.
            const TF esl = esat_liq(T);
            const TF esi = esat_ice(T);

            // Tomita Eq. 57
            const TF G_w = TF(1.) / (
                Lv<TF> / (K_a<TF> * T) * (Lv<TF> / (Rv<TF> * T) - TF(1.))
                + Rv<TF>*T / (K_d<TF> * esl) );

            // Tomita Eq. 62
            const TF G_i = TF(1.) / (
                Ls<TF> / (K_a<TF> * T) * (Ls<TF> / (Rv<TF> * T) - TF(1.))
                + Rv<TF>*T / (K_d<TF> * esi) );

            // qsat_liq and qsat_ice, reusing esat.
            const TF qsl = ep<TF>*esl / (p_k - (TF(1.)-ep<TF>)*esl);
            const TF qsi = ep<TF>*esi / (p_k - (TF(1.)-ep<TF>)*esi);

            const TF S_w = qv / qsl;
            const TF S_i = qv / qsi;

            // Tomita Eq. 63
            const TF delta_3 = TF(S_i <= TF(1.)); // Subsaturated, then delta_3 = 1.

            // Ventilation terms of rain, snow and graupel.
            const TF vent_r = vent_1r * lr_2 + vent_2r * lr_2 * lr_12 * lr_14;
            const TF vent_s = vent_1s * ls_2 + vent_2s * ls_2 * ls_12 * ls_18;
            const TF vent_g = vent_1g * lg_2 + vent_2g * lg_2 * lg_12 * lg_14;

            // Tomita Eq. 59
            TF P_revp = !(has_rain) ? TF(0.) :
                - TF(2.)*pi<TF> * N_0r<TF> * (std::min(S_w, TF(1.)) - TF(1.)) * G_w * rho_i * vent_r;

            // Tomita Eq. 60. Negative for sublimation, positive for deposition.
            const TF P_sdep_ssub =
                TF(2.)*pi<TF> * N_0s<TF> * (S_i - TF(1.)) * G_i * rho_i * vent_s;

            // Tomita Eq. 51
            const TF P_gdep_gsub =
                TF(2.)*pi<TF> * N_0g<TF> * (S_i - TF(1.)) * G_i * rho_i * vent_g;

            // Tomita Eq. 64
            TF P_sdep = !(has_vapor) ? TF(0.) :
                (TF(1.) - delta_3) * P_sdep_ssub;
            TF P_gdep = !(has_vapor) ? TF(0.) :
                (TF(1.) - delta_3) * P_gdep_gsub;

            // Tomita Eq. 65
            // CvH: I swapped the sign with respect to Tomita, the term should be positive.
            TF P_ssub = !(has_snow) ? TF(0.) :
                - delta_3 * P_sdep_ssub;
            TF P_gsub = !(has_graupel) ? TF(0.) :
                - delta_3 * P_gdep_gsub;

            // Freezing and melting
            // Tomita Eq. 67, 68 combined.
            TF P_smlt = !(has_snow) ? TF(0.) :
                TF(2.)*pi<TF> * K_a<TF> * (T - T0<TF>) * N_0s<TF> * rho_i / Lf<TF> * vent_s
                + C_l<TF> * (T - T0<TF>) / Lf<TF> * (P_sacw + P_sacr);

            // Tomita Eq. 69
            TF P_gmlt = !(has_graupel) ? TF(0.) :
                TF(2.)*pi<TF> * K_a<TF> * (T - T0<TF>) * N_0g<TF> * rho_i / Lf<TF> * vent_g
                + C_l<TF> * (T - T0<TF>) / Lf<TF> * (P_gacw + P_gacr);

            // Tomita Eq. 70
            constexpr TF A_prime = TF(0.66);
            constexpr TF B_prime = TF(100.);

            TF P_gfrz = !(has_rain) ? TF(0.) :
                TF(20.) * pi_2<TF> * B_prime * N_0r<TF> * rho_w<TF> * rho_i
                * (std::exp(A_prime * (T0<TF> - T)) - TF(1.)) * pow7(lambda_r_i);

            // COMPUTE THE TENDENCIES.
            // Limit the production terms to avoid instability.
            auto limit_tend = [&](TF& tend, const TF tend_limit)
            {
                tend = std::max(TF(0.), std::min(tend, tend_limit));
            };

            const TF dqv_dt_max = qv      / dt;
            const TF dqi_dt_max = qi_c[n] / dt;
            const TF dql_dt_max = ql_c[n] / dt;
            const TF dqr_dt_max = qr_c[n] / dt;
            const TF dqs_dt_max = qs_c[n] / dt;
            const TF dqg_dt_max = qg_c[n] / dt;

            // Limit on the availability of the source.
            // Limit accretion terms.
            limit_tend(P_iacr_s, dqr_dt_max);
            limit_tend(P_iacr_g, dqr_dt_max);
            limit_tend(P_raci_s, dqi_dt_max);
            limit_tend(P_raci_g, dqi_dt_max);
            limit_tend(P_racw  , dql_dt_max);
            limit_tend(P_sacw  , dql_dt_max);
            limit_tend(P_saci  , dqi_dt_max);
            limit_tend(P_gacw  , dql_dt_max);
            limit_tend(P_gaci  , dqi_dt_max);
            limit_tend(P_racs  , dqs_dt_max);
            limit_tend(P_sacr_s, dqr_dt_max);
            limit_tend(P_sacr_g, dqr_dt_max);
            limit_tend(P_gacr  , dqr_dt_max);
            limit_tend(P_gacs  , dqs_dt_max);

            // Limit autoconversion terms.
            limit_tend(P_raut, dql_dt_max);
            limit_tend(P_saut, dqi_dt_max);
            limit_tend(P_gaut, dqs_dt_max);

            // Limit phase changes.
            limit_tend(P_revp, dqr_dt_max);
            limit_tend(P_sdep, dqv_dt_max);
            limit_tend(P_ssub, dqs_dt_max);
            limit_tend(P_gdep, dqv_dt_max);
            limit_tend(P_gsub, dqg_dt_max);
            limit_tend(P_smlt, dqs_dt_max);
            limit_tend(P_gmlt, dqg_dt_max);
            limit_tend(P_gfrz, dqr_dt_max);

            // P_iacr_s = 0;
            // P_iacr_g = 0;
            // P_raci_s = 0;
            // P_raci_g = 0;
            // P_racw   = 0;
            // P_sacw   = 0;
            // P_saci   = 0;
            // P_gacw   = 0;
            // P_gaci   = 0;
            // P_racs   = 0;
            // P_sacr_s = 0;
            // P_sacr_g = 0;
            // P_gacr   = 0;
            // P_gacs   = 0;

            // P_raut = 0;
            // P_saut = 0;
            // P_gaut = 0;

            // P_revp = 0;
            // P_sdep = 0;
            // P_ssub = 0;
            // P_gdep = 0;
            // P_gsub = 0;
            // P_smlt = 0;
            // P_gmlt = 0;
            // P_gfrz = 0;

            // MT: for the * T_pos and * T_neg I followed the table in the SCALE documentation.
            // I assumed that has_ice already guarantees negative temperatures.
            TF vapor_to_snow = P_sdep;
            TF vapor_to_graupel = P_gdep;

            TF cloud_to_rain = P_racw + P_sacw * T_pos + P_raut + P_gacw * T_pos;
            TF cloud_to_graupel = P_gacw * T_neg;
            TF cloud_to_snow = P_sacw * T_neg;

            TF rain_to_vapor = P_revp;
            TF rain_to_graupel = P_gacr * T_neg + P_iacr_g + P_sacr_g * T_neg + P_gfrz * T_neg;
            TF rain_to_snow = P_sacr_s * T_neg + P_iacr_s;

            TF ice_to_snow = P_raci_s + P_saci + P_saut;
            TF ice_to_graupel = P_raci_g + P_gaci;

            TF snow_to_graupel = P_gacs + P_racs * T_neg + P_gaut * T_neg;
            TF snow_to_rain = P_smlt * T_pos;
            TF snow_to_vapor = P_ssub;

            TF graupel_to_rain = P_gmlt * T_pos;
            TF graupel_to_vapor = P_gsub;

            const TF dqv_dt =
                - vapor_to_snow - vapor_to_graupel + snow_to_vapor + graupel_to_vapor + rain_to_vapor;

            const TF dql_dt =
                - cloud_to_rain - cloud_to_graupel - cloud_to_snow;

            const TF dqi_dt =
                - ice_to_snow - ice_to_graupel;

            const TF dqr_dt =
                + cloud_to_rain + snow_to_rain + graupel_to_rain
                - rain_to_vapor - rain_to_graupel - rain_to_snow;

            const TF dqs_dt =
                + cloud_to_snow + ice_to_snow + vapor_to_snow
                - snow_to_graupel - snow_to_vapor - snow_to_rain + rain_to_snow;

            const TF dqg_dt =
                + cloud_to_graupel + rain_to_graupel + ice_to_graupel
                + vapor_to_graupel + snow_to_graupel
                - graupel_to_rain - graupel_to_vapor;

            // Limit the production terms to avoid instability.
            auto limit_factor = [](const TF tend, const TF tend_limit)
            {
                return (tend < TF(0.)) ? std::min(-tend_limit/tend, TF(1.)) : TF(1.);
            };

            const TF dqv_dt_fac = limit_factor(dqv_dt, dqv_dt_max);
            const TF dql_dt_fac = limit_factor(dql_dt, dql_dt_max);
            const TF dqi_dt_fac = limit_factor(dqi_dt, dqi_dt_max);
            const TF dqr_dt_fac = limit_factor(dqr_dt, dqr_dt_max);
            const TF dqs_dt_fac = limit_factor(dqs_dt, dqs_dt_max);
            const TF dqg_dt_fac = limit_factor(dqg_dt, dqg_dt_max);

            vapor_to_snow    *= dqv_dt_fac * dqs_dt_fac;
            vapor_to_graupel *= dqv_dt_fac * dqg_dt_fac;

            cloud_to_rain    *= dql_dt_fac * dqr_dt_fac;
            cloud_to_graupel *= dql_dt_fac * dqg_dt_fac;
            cloud_to_snow    *= dql_dt_fac * dqs_dt_fac;

            rain_to_vapor    *= dqr_dt_fac * dqv_dt_fac;
            rain_to_graupel  *= dqr_dt_fac * dqg_dt_fac;
            rain_to_snow     *= dqr_dt_fac * dqs_dt_fac;

            ice_to_snow      *= dqi_dt_fac * dqs_dt_fac;
            ice_to_graupel   *= dqi_dt_fac * dqg_dt_fac;

            snow_to_graupel  *= dqs_dt_fac * dqg_dt_fac;
            snow_to_vapor    *= dqs_dt_fac * dqv_dt_fac;
            snow_to_rain     *= dqs_dt_fac * dqr_dt_fac;

            graupel_to_rain  *= dqg_dt_fac * dqr_dt_fac;
            graupel_to_vapor *= dqg_dt_fac * dqv_dt_fac;

            frz_c[n] =
                  cloud_to_graupel + cloud_to_snow + rain_to_graupel + rain_to_snow
                + vapor_to_snow + vapor_to_graupel
                - snow_to_rain - graupel_to_rain - snow_to_vapor - graupel_to_vapor;

            /* Conversion of the process rates to thl and qt tendencies (thlE).
             * thlE is not linear in T, ql and qi, so the thl tendency cannot be summed per process
             * (thlD: thlt += L/(cp*exner) * rate). Instead:
             *   1. accumulate the changes in cloud liquid (dqc), cloud ice (dqi) and vapor (dqv),
             *   2. compute the latent heating dT of all phase changes at constant cp,
             *   3. convert T_end, qc_end, qi_end to thl_end with thlE, and
             *   4. thlt += (thl_end - thl) / dt and qtt += dqc + dqi + dqv.
             * Precipitation (qr, qs, qg) is not part of thl or qt; their tendencies are added directly.
             */
            const TF dqc = - cloud_to_rain - cloud_to_graupel - cloud_to_snow;
            const TF dqi = - ice_to_snow - ice_to_graupel;
            const TF dqv = - vapor_to_snow - vapor_to_graupel + rain_to_vapor + snow_to_vapor + graupel_to_vapor;

            dqr_c[n] =
                + cloud_to_rain - rain_to_vapor - rain_to_graupel - rain_to_snow
                + snow_to_rain + graupel_to_rain;
            dqs_c[n] =
                + vapor_to_snow + cloud_to_snow + rain_to_snow + ice_to_snow
                - snow_to_graupel - snow_to_vapor - snow_to_rain;
            dqg_c[n] =
                + vapor_to_graupel + cloud_to_graupel + rain_to_graupel + ice_to_graupel
                + snow_to_graupel - graupel_to_rain - graupel_to_vapor;
            dqt_c[n] = dqc + dqi + dqv;

            // State after one time step of the microphysics.
            const TF qc_end = ql_c[n] + dqc * dt;
            const TF qi_end = qi_c[n] + dqi * dt;

            // Latent heating [K s-1] of all phase changes, with constant Lv/Ls/Lf and cp (consistent with thlE).
            // Transfers within one phase (cloud_to_rain, ice_to_snow, ...) do not change T.
            const TF dT =   - Lv<TF> / cp<TF> * rain_to_vapor
                            - Ls<TF> / cp<TF> * snow_to_vapor
                            - Ls<TF> / cp<TF> * graupel_to_vapor
                            + Lf<TF> / cp<TF> * cloud_to_snow
                            + Lf<TF> / cp<TF> * cloud_to_graupel
                            - Lf<TF> / cp<TF> * snow_to_rain
                            - Lf<TF> / cp<TF> * graupel_to_rain
                            + Lf<TF> / cp<TF> * rain_to_snow
                            + Lf<TF> / cp<TF> * rain_to_graupel
                            + Ls<TF> / cp<TF> * vapor_to_graupel
                            + Ls<TF> / cp<TF> * vapor_to_snow;

            const TF T_end = T + dT * dt;

            // Convert the start and end state to thl: thlE = T/exner / (1 + Lv*ql/(cp*T) + Ls*qi/(cp*T)).
            // The start state is recomputed from T instead of using thl, such that cells without
            // conversions get an exactly zero tendency instead of the round-off of the T inversion.
            const TF thl_start = T/exner_k / (1 + Lv<TF>*ql_c[n]/(cp<TF> * T)
                                                + Ls<TF>*qi_c[n]/(cp<TF> * T));
            const TF thl_end = T_end/exner_k / (1 + Lv<TF>*qc_end/(cp<TF> * T_end)
                                                   + Ls<TF>*qi_end/(cp<TF> * T_end));

            // Tendencies of the conserved variables.
            dthl_c[n] = (thl_end - thl_start)/dt;
        }
    }

    // Compute all microphysical tendencies.
    template<typename TF>
    void conversion(
            TF* const restrict qrt, TF* const restrict qst, TF* const restrict qgt,
            TF* const restrict qtt, TF* const restrict thlt,
            TF* const restrict net_frz,
            TF* const restrict thl_tend_micro,
            TF* const restrict qt_tend_micro,
            const TF* const restrict qr, const TF* const restrict qs, const TF* const restrict qg,
            const TF* const restrict qt, const TF* const restrict thl,
            const TF* const restrict ql, const TF* const restrict qi,
            const TF* const restrict rho, const TF* const restrict exner, const TF* const restrict p,
            const TF* const restrict dzi, const TF* const restrict dzhi,
            const TF Nc0, const TF dt,
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            const int jj, const int kk)
    {
        // Tomita Eq. 51. Nc0 is converted from SI units (m-3 instead of cm-3).
        const TF D_d = TF(0.146) - TF(5.964e-2)*std::log((Nc0*TF(1.e-6)) / TF(2.e3));

        // The powers of lambda below are written out for these exponents.
        static_assert(b_r<TF> == TF(3.) && b_s<TF> == TF(3.) && b_g<TF> == TF(3.), "nsw6 assumes b = 3");
        static_assert(d_r<TF> == TF(0.5) && d_s<TF> == TF(0.25) && d_g<TF> == TF(0.5), "nsw6 assumes d_r = d_g = 0.5, d_s = 0.25");

        // Tomita Eq. 27: 1/lambda = (rho*q / (a*N_0*gamma(b+1)))^(1/(b+1)).
        const TF lambda_fac_r = TF(1.) / (a_r<TF> * N_0r<TF> * std::tgamma(b_r<TF> + TF(1.)));
        const TF lambda_fac_s = TF(1.) / (a_s<TF> * N_0s<TF> * std::tgamma(b_s<TF> + TF(1.)));
        const TF lambda_fac_g = TF(1.) / (a_g<TF> * N_0g<TF> * std::tgamma(b_g<TF> + TF(1.)));

        // Tomita Eq. 28, without the density correction.
        const TF V_T_fac_r = c_r<TF> * std::tgamma(b_r<TF> + d_r<TF> + TF(1.)) / std::tgamma(b_r<TF> + TF(1.));
        const TF V_T_fac_s = c_s<TF> * std::tgamma(b_s<TF> + d_s<TF> + TF(1.)) / std::tgamma(b_s<TF> + TF(1.));
        const TF V_T_fac_g = c_g<TF> * std::tgamma(b_g<TF> + d_g<TF> + TF(1.)) / std::tgamma(b_g<TF> + TF(1.));

        // Tomita Eq. 29, 32 - 38, without the density correction.
        const TF fac_iacr0 = pi_2<TF> * E_ri<TF> * N_0r<TF> * c_r<TF> * rho_w<TF> * std::tgamma(TF(6.) + d_r<TF>) / (TF(24.) * M_i<TF>);
        const TF fac_raci0 = pi<TF> * E_ri<TF> * N_0r<TF> * c_r<TF> * std::tgamma(TF(3.) + d_r<TF>) / TF(4.);
        const TF fac_racw0 = pi<TF> * E_rw<TF> * N_0r<TF> * c_r<TF> * std::tgamma(TF(3.) + d_r<TF>) / TF(4.);
        const TF fac_sacw0 = pi<TF> * E_sw<TF> * N_0s<TF> * c_s<TF> * std::tgamma(TF(3.) + d_s<TF>) / TF(4.);
        const TF fac_saci0 = pi<TF> * N_0s<TF> * c_s<TF> * std::tgamma(TF(3.) + d_s<TF>) / TF(4.);
        const TF fac_gacw0 = pi<TF> * E_gw<TF> * N_0g<TF> * c_g<TF> * std::tgamma(TF(3.) + d_g<TF>) / TF(4.);
        const TF fac_gaci0 = pi<TF> * E_gi<TF> * N_0g<TF> * c_g<TF> * std::tgamma(TF(3.) + d_g<TF>) / TF(4.);

        // Tomita Eq. 41, 44, 47, 48, without the 1/rho and E_gs.
        const TF fac_racs0 = pi<TF> * a_s<TF> * E_sr<TF> * N_0s<TF> * N_0r<TF> / TF(4.);
        const TF fac_sacr0 = pi<TF> * a_r<TF> * E_sr<TF> * N_0r<TF> * N_0s<TF> / TF(4.);
        const TF fac_gacr0 = pi<TF> * a_r<TF> * E_gr<TF> * N_0g<TF> * N_0r<TF> / TF(4.);
        const TF fac_gacs0 = pi<TF> * a_s<TF> * N_0g<TF> * N_0s<TF> / TF(4.);

        // Gamma function weights of the collection integrals in Eq. 41, 44, 47, 48.
        const TF g_s1 = std::tgamma(b_s<TF> + TF(1.)) * std::tgamma(TF(3.));
        const TF g_s2 = TF(2.) * std::tgamma(b_s<TF> + TF(2.)) * std::tgamma(TF(2.));
        const TF g_s3 = std::tgamma(b_s<TF> + TF(3.)) * std::tgamma(TF(1.));
        const TF g_r1 = std::tgamma(b_r<TF> + TF(1.)) * std::tgamma(TF(3.));
        const TF g_r2 = TF(2.) * std::tgamma(b_r<TF> + TF(2.)) * std::tgamma(TF(2.));
        const TF g_r3 = std::tgamma(b_r<TF> + TF(3.)) * std::tgamma(TF(1.));

        // Ventilation factors of Eq. 59, 60, 61, 67 - 69, without the density correction.
        const TF vent_1r = f_1r<TF> * std::tgamma(TF(2.));
        const TF vent_1s = f_1s<TF> * std::tgamma(TF(2.));
        const TF vent_1g = f_1g<TF> * std::tgamma(TF(2.));
        const TF vent_2r0 = f_2r<TF> * std::sqrt(c_r<TF> / nu<TF>) * std::tgamma(TF(0.5) * (TF(5.) + d_r<TF>));
        const TF vent_2s0 = f_2s<TF> * std::sqrt(c_s<TF> / nu<TF>) * std::tgamma(TF(0.5) * (TF(5.) + d_s<TF>));
        const TF vent_2g0 = f_2g<TF> * std::sqrt(c_g<TF> / nu<TF>) * std::tgamma(TF(0.5) * (TF(5.) + d_g<TF>));

        // Kharoutdinov and Kogan autoconversion prefactor.
        const TF fac_raut = TF(1350.) * std::pow(Nc0 * TF(1.e-6), TF(-1.79));

        // Per height level, the cells that contain condensate or precipitation are compacted into
        // a list, which is processed in blocks of contiguous arrays to allow vectorization.
        constexpr int nblock = 256;
        std::vector<int> index((iend-istart)*(jend-jstart));
        std::vector<int> is_active_row(iend-istart);

        enum Block_vars {
            qr_b, qs_b, qg_b, ql_b, qi_b, qt_b, thl_b,
            dqr_b, dqs_b, dqg_b, dqt_b, dthl_b, frz_b,
            n_block_vars };

        std::vector<TF> block_data(n_block_vars*nblock);
        auto block = [&](const Block_vars v) { return block_data.data() + v*nblock; };

        TF* const restrict qr_c  = block(qr_b);
        TF* const restrict qs_c  = block(qs_b);
        TF* const restrict qg_c  = block(qg_b);
        TF* const restrict ql_c  = block(ql_b);
        TF* const restrict qi_c  = block(qi_b);
        TF* const restrict qt_c  = block(qt_b);
        TF* const restrict thl_c = block(thl_b);

        TF* const restrict dqr_c  = block(dqr_b);
        TF* const restrict dqs_c  = block(dqs_b);
        TF* const restrict dqg_c  = block(dqg_b);
        TF* const restrict dqt_c  = block(dqt_b);
        TF* const restrict dthl_c = block(dthl_b);
        TF* const restrict frz_c  = block(frz_b);

        for (int k=kstart; k<kend; ++k)
        {
            const TF rho0_rho_sqrt = std::sqrt(rho[kstart]/rho[k]);
            const TF rho0_rho_4th = std::sqrt(rho0_rho_sqrt);
            const TF rho_k = rho[k];
            const TF rho_i = TF(1.) / rho[k];
            const TF exner_k = exner[k];
            const TF p_k = p[k];

            const TF fac_iacr = fac_iacr0 * rho0_rho_sqrt;
            const TF fac_raci = fac_raci0 * rho0_rho_sqrt;
            const TF fac_racw = fac_racw0 * rho0_rho_sqrt;
            const TF fac_sacw = fac_sacw0 * rho0_rho_sqrt;
            const TF fac_saci = fac_saci0 * rho0_rho_sqrt;
            const TF fac_gacw = fac_gacw0 * rho0_rho_sqrt;
            const TF fac_gaci = fac_gaci0 * rho0_rho_sqrt;

            const TF V_T_fac_r_k = V_T_fac_r * rho0_rho_sqrt;
            const TF V_T_fac_s_k = V_T_fac_s * rho0_rho_sqrt;
            const TF V_T_fac_g_k = V_T_fac_g * rho0_rho_sqrt;

            const TF vent_2r = vent_2r0 * rho0_rho_4th;
            const TF vent_2s = vent_2s0 * rho0_rho_4th;
            const TF vent_2g = vent_2g0 * rho0_rho_4th;

            const Conversion_constants<TF> cst = {
                lambda_fac_r,
                lambda_fac_s,
                lambda_fac_g,
                V_T_fac_r_k,
                V_T_fac_s_k,
                V_T_fac_g_k,
                fac_iacr,
                fac_raci,
                fac_racw,
                fac_sacw,
                fac_saci,
                fac_gacw,
                fac_gaci,
                fac_racs0,
                fac_sacr0,
                fac_gacr0,
                fac_gacs0,
                g_s1,
                g_s2,
                g_s3,
                g_r1,
                g_r2,
                g_r3,
                vent_1r,
                vent_1s,
                vent_1g,
                vent_2r,
                vent_2s,
                vent_2g,
                fac_raut,
                rho_k,
                rho_i,
                exner_k,
                p_k };

            // Compact the cells with condensate or precipitation.
            int n_cells = 0;
            for (int j=jstart; j<jend; ++j)
            {
                // Flag the active cells of one row in a vectorizable loop, then compact the row.
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    is_active_row[i-istart] =
                          int(ql[ijk] > ql_min<TF>) | int(qi[ijk] > qi_min<TF>)
                        | int(qr[ijk] > qr_min<TF>) | int(qs[ijk] > qs_min<TF>) | int(qg[ijk] > qg_min<TF>);
                }

                for (int i=istart; i<iend; ++i)
                {
                    index[n_cells] = i + j*jj + k*kk;
                    n_cells += is_active_row[i-istart];
                }
            }

            for (int n_start=0; n_start<n_cells; n_start+=nblock)
            {
                const int nb = std::min(nblock, n_cells-n_start);
                const int* const restrict ids = index.data() + n_start;

                // Gather.
                for (int n=0; n<nb; ++n)
                {
                    const int ijk = ids[n];
                    qr_c[n]  = qr[ijk];
                    qs_c[n]  = qs[ijk];
                    qg_c[n]  = qg[ijk];
                    ql_c[n]  = ql[ijk];
                    qi_c[n]  = qi[ijk];
                    qt_c[n]  = qt[ijk];
                    thl_c[n] = thl[ijk];
                }

                conversion_block(
                        dqr_c, dqs_c, dqg_c, dqt_c, dthl_c, frz_c,
                        qr_c, qs_c, qg_c, ql_c, qi_c, qt_c, thl_c,
                        cst, dt, nb);

                // Scatter.
                for (int n=0; n<nb; ++n)
                {
                    const int ijk = ids[n];
                    qrt[ijk]  += dqr_c[n];
                    qst[ijk]  += dqs_c[n];
                    qgt[ijk]  += dqg_c[n];
                    qtt[ijk]  += dqt_c[n];
                    thlt[ijk] += dthl_c[n];
                }

                if (net_frz != nullptr)
                    for (int n=0; n<nb; ++n)
                        net_frz[ids[n]] = frz_c[n];

                if (thl_tend_micro != nullptr)
                    for (int n=0; n<nb; ++n)
                        thl_tend_micro[ids[n]] = dthl_c[n];

                if (qt_tend_micro != nullptr)
                    for (int n=0; n<nb; ++n)
                        qt_tend_micro[ids[n]] = dqt_c[n];
            }
        }
    }

    // Bergeron.
    template<typename TF>
    void bergeron(
            TF* const restrict qst,
            TF* const restrict qtt, TF* const restrict thlt,
            const TF* const restrict ql, const TF* const restrict qi,
            const TF* const restrict rho, const TF* const restrict exner,
            const TF delta_t,
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            const int jj, const int kk)
    {
        constexpr TF m_i40 = TF(2.46e-10);
        constexpr TF m_i50 = TF(4.8e-10);
        constexpr TF R_i50 = TF(5.e-5);

        // constexpr TF a1 = 
        // constexpr TF a2 = 

        // constexpr TF delta_t1 =
        //     ( std::pow(m_i50, TF(1.) - a_2) - std::pow(m_i40, TF(1.) - a_2) )
        //     / (a_1 * (TF(1.) - a_2));

        for (int k=kstart; k<kend; k++)
            for (int j=jstart; j<jend; j++)
                #pragma ivdep
                for (int i=istart; i<iend; i++)
                {
                    const int ijk = i + j*jj + k*kk;
                    // To be filled in.
                }
    }

    // Highest level with hydrometeors that sediment (qc > qc_min), or with negative qc. Above k_top + 1,
    // the sedimentation velocity, CFL number and flux are exactly zero. Returns kstart-1 if there are none.
    template<typename TF>
    int calc_k_top(
            const TF* const restrict qc, const TF qc_min,
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            const int jj, const int kk)
    {
        for (int k=kend-1; k>=kstart; --k)
        {
            int n_active = 0;
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    n_active += int(qc[ijk] > qc_min) | int(qc[ijk] < TF(0.));
                }

            if (n_active > 0)
                return k;
        }

        return kstart-1;
    }

    // Terminal velocity (Tomita Eq. 27, 28) at cell centers of levels kstart to kend_s, limited between 0.1 and 10 m/s.
    template<typename TF>
    void calc_terminal_velocity(
            TF* const restrict w_qc,
            const TF* const restrict qc,
            const TF* const restrict rho,
            const TF a_c, const TF b_c, const TF c_c, const TF d_c, const TF N_0c,
            const TF qc_min,
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend_s,
            const int jj, const int kk)
    {
        constexpr TF V_Tmin = TF(0.1);
        constexpr TF V_Tmax = TF(10.);

        // V_T = c * sqrt(rho0/rho) * gamma(b+d+1)/gamma(b+1) * lambda^-d,
        // with lambda^-d = (rho*q / (a*N_0*gamma(b+1)))^(d/(b+1)).
        const TF V_T_fac = c_c * std::tgamma(b_c + d_c + TF(1.)) / std::tgamma(b_c + TF(1.));
        const TF lambda_fac = TF(1.) / (a_c * N_0c * std::tgamma(b_c + TF(1.)));
        const TF lambda_exp = d_c / (b_c + TF(1.));

        auto calc_w = [&](auto lambda_pow)
        {
            for (int k=kstart; k<kend_s; ++k)
            {
                const TF V_T_fac_k = V_T_fac * std::sqrt(rho[kstart]/rho[k]);
                const TF lambda_fac_k = lambda_fac * rho[k];

                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;

                        // Written as branch, such that it is skipped in empty cells if the compiler
                        // does not vectorize this loop (without -ffast-math).
                        TF w = TF(0.);
                        if (qc[ijk] > qc_min)
                        {
                            const TF V_T = V_T_fac_k * lambda_pow(lambda_fac_k * qc[ijk]);
                            w = std::max(V_Tmin, std::min(V_T, V_Tmax));
                        }
                        w_qc[ijk] = w;
                    }
            }
        };

        // Square roots are much cheaper than pow for the exponents of rain, graupel (1/8) and snow (1/16).
        if (lambda_exp == TF(0.125))
            calc_w([](const TF x) { return std::sqrt(std::sqrt(std::sqrt(x))); });
        else if (lambda_exp == TF(0.0625))
            calc_w([](const TF x) { return std::sqrt(std::sqrt(std::sqrt(std::sqrt(x)))); });
        else
            calc_w([&](const TF x) { return std::pow(x, lambda_exp); });

        // Set the ghost cells.
        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ijk_bot = i + j*jj + (kstart-1)*kk;
                const int ijk_top = i + j*jj + kend_s*kk;
                w_qc[ijk_bot] = w_qc[ijk_bot+kk];
                w_qc[ijk_top] = TF(0.);
            }
    }

    // Sedimentation based on Stevens and Seifert (2008)
    template<typename TF>
    void sedimentation_ss08(
            TF* const restrict qct, TF* const restrict rc_bot,
            TF* const restrict w_qc, TF* const restrict c_qc,
            TF* const restrict slope_qc, TF* const restrict flux_qc,
            TF* const restrict ftot_v, TF* const restrict dzz_v, TF* const restrict cc_v,
            const TF* const restrict qc,
            const TF* const restrict rho,
            const TF* const restrict dzi, const TF* const restrict dz,
            const double dt,
            const TF a_c, const TF b_c, const TF c_c, const TF d_c, const TF N_0c,
            const TF qc_min,
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            const int jj, const int kk)
    {
        // Time step in working precision, to avoid mixed precision in the loops.
        const TF dt_tf = TF(dt);

        // All computations are limited to the levels below kend_s, above which everything is zero.
        const int k_top = calc_k_top(qc, qc_min, istart, jstart, kstart, iend, jend, kend, jj, kk);
        const int kend_s = std::min(k_top+3, kend);

        // 1. Calculate sedimentation velocity at cell center
        calc_terminal_velocity(
                w_qc, qc, rho,
                a_c, b_c, c_c, d_c, N_0c, qc_min,
                istart, jstart, kstart,
                iend, jend, kend_s,
                jj, kk);

        // 2. Calculate CFL number using interpolated sedimentation velocity, and 3. calculate slopes
        for (int k=kstart; k<kend_s; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    c_qc[ijk] = TF(0.25) * (w_qc[ijk-kk] + TF(2.)*w_qc[ijk] + w_qc[ijk+kk]) * dzi[k] * dt_tf;
                    slope_qc[ijk] = minmod(qc[ijk]-qc[ijk-kk], qc[ijk+kk]-qc[ijk]);
                }

        // Calculate flux
        // Set the fluxes at the top of the domain (kend_s) to zero
        for (int j=jstart; j<jend; ++j)
            for (int i=istart; i<iend; ++i)
            {
                const int ijk = i + j*jj + kend_s*kk;
                flux_qc[ijk] = TF(0.);
            }

        // Per level, the integration of the flux over the levels above is done for all columns
        // in lockstep, with the state of each column in the 2D fields ftot_v, dzz_v and cc_v.

        for (int k=kend_s-1; k>kstart-1; --k)
        {
            // q_rain
            int n_active = 0;
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij = i + j*jj;
                    const int ijk = ij + k*kk;
                    ftot_v[ij] = TF(0.); // cumulative 'flux' (kg m-2)
                    dzz_v[ij] = TF(0.); // distance from zh[k]
                    cc_v[ij] = std::min(TF(1.), c_qc[ijk]);
                    n_active += cc_v[ij] > TF(0.);
                }

            for (int kc=k; kc<kend && n_active>0; ++kc) // kc = current grid level
            {
                n_active = 0;
                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    #pragma clang loop vectorize(enable)
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ij = i + j*jj;
                        const int ijkc = ij + kc*kk;

                        const TF cc = cc_v[ij];
                        const bool active = cc > TF(0.);

                        const TF ftot_new = ftot_v[ij] + rho[kc] * (qc[ijkc] + TF(0.5) * slope_qc[ijkc] * (TF(1.)-cc)) * cc * dz[kc];
                        const TF dzz_new = dzz_v[ij] + dz[kc];
                        const TF cc_new = std::min(TF(1.), c_qc[ijkc] - dzz_new*dzi[kc+1]);

                        ftot_v[ij] = active ? ftot_new : ftot_v[ij];
                        dzz_v[ij] = active ? dzz_new : dzz_v[ij];
                        cc_v[ij] = active ? cc_new : cc;
                        n_active += int(active) & int(cc_new > TF(0.));
                    }
            }

            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij = i + j*jj;
                    const int ijk = ij + k*kk;

                    // Given flux at top, limit bottom flux such that the total rain content stays >= 0.
                    const TF ftot = std::min(ftot_v[ij], rho[k] * dz[k] * qc[ijk] - flux_qc[ijk+kk] * dt_tf);
                    flux_qc[ijk] = -ftot / dt_tf;

                    // Calculate tendency
                    qct[ijk] += -(flux_qc[ijk+kk] - flux_qc[ijk]) / rho[k] * dzi[k];
                }
        }

        // Store surface sedimentation flux
        // Sedimentation flux is already multiplied with density (see flux div. calculation), so
        // the resulting flux is in kg m-2 s-1, with rho_water = 1000 kg/m3 this equals a rain rate in mm s-1
        for (int j=jstart; j<jend; ++j)
            for (int i=istart; i<iend; ++i)
            {
                const int ij  = i + j*jj;
                const int ijk = i + j*jj + kstart*kk;

                rc_bot[ij] = -flux_qc[ijk];
            }
    }

    // Sedimentation from Stevens and Seifert (2008)
    template<typename TF>
    TF calc_cfl_ss08(
            TF* const restrict w_qc,
            const TF* const restrict qc,
            const TF* const restrict rho,
            const TF* const restrict dzi, const TF* const restrict dz,
            const double dt,
            const TF a_c, const TF b_c, const TF c_c, const TF d_c, const TF N_0c,
            const TF qc_min,
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            const int jj, const int kk)
    {
        TF cfl_max = TF(0.);

        // Above kend_s, the CFL number is exactly zero.
        const int k_top = calc_k_top(qc, qc_min, istart, jstart, kstart, iend, jend, kend, jj, kk);
        const int kend_s = std::min(k_top+3, kend);

        // 1. Calculate sedimentation velocity at cell center
        calc_terminal_velocity(
                w_qc, qc, rho,
                a_c, b_c, c_c, d_c, N_0c, qc_min,
                istart, jstart, kstart,
                iend, jend, kend_s,
                jj, kk);

        // 2. Calculate CFL number using interpolated sedimentation velocity
        for (int k=kstart; k<kend_s; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    const TF cfl = TF(0.25) * (w_qc[ijk-kk] + TF(2.)*w_qc[ijk] + w_qc[ijk+kk]) * dzi[k] * TF(dt);
                    cfl_max = std::max(cfl, cfl_max);
                }

        return cfl_max;
    }

    // Sum the individual surface precipitation rates into one combined rate.
    template<typename TF>
    void calc_rrsg_bot(
            TF* const restrict rrsg_bot,
            const TF* const restrict rr_bot,
            const TF* const restrict rs_bot,
            const TF* const restrict rg_bot,
            const int ncells)
    {
        for (int n=0; n<ncells; ++n)
            rrsg_bot[n] = rr_bot[n] + rs_bot[n] + rg_bot[n];
    }
}

template<typename TF>
Microphys_nsw6<TF>::Microphys_nsw6(Master& masterin, Grid<TF>& gridin, Fields<TF>& fieldsin, Input& inputin) :
    Microphys<TF>(masterin, gridin, fieldsin, inputin)
{
    auto& gd = grid.get_grid_data();
    swmicrophys = Microphys_type::Nsw6;

    // Read microphysics switches and settings
    // swmicrobudget = inputin.get_item<bool>("micro", "swmicrobudget", "", false);
    cflmax = inputin.get_item<TF>("micro", "cflmax", "", 1.2);
    Nc0 = inputin.get_item<TF>("micro", "Nc0", "");

    // Initialize the qr (rain water specific humidity) and nr (droplot number concentration) fields
    const std::string group_name = "thermo";

    fields.init_prognostic_field("qr", "Rain water specific humidity", "kg kg-1", group_name, gd.sloc, false);
    fields.init_prognostic_field("qs", "Snow specific humidity", "kg kg-1", group_name, gd.sloc, false);
    fields.init_prognostic_field("qg", "Graupel specific humidity", "kg kg-1", group_name, gd.sloc, false);

    // Load the viscosity for both fields.
    fields.sp.at("qr")->visc = inputin.get_item<TF>("fields", "svisc", "qr");
    fields.sp.at("qg")->visc = inputin.get_item<TF>("fields", "svisc", "qg");
    fields.sp.at("qs")->visc = inputin.get_item<TF>("fields", "svisc", "qs");
}

template<typename TF>
Microphys_nsw6<TF>::~Microphys_nsw6()
{
}

template<typename TF>
void Microphys_nsw6<TF>::init(Thermo<TF>& thermo)
{
    auto& gd = grid.get_grid_data();

    rr_bot.resize(gd.ijcells);
    rs_bot.resize(gd.ijcells);
    rg_bot.resize(gd.ijcells);

    std::vector<std::string>& hydrometeorlist = thermo.get_hydrometeorlist();
    hydrometeorlist = {"qr", "qs", "qg"};
}

template<typename TF>
void Microphys_nsw6<TF>::create(
        Input& inputin, Netcdf_handle& input_nc,
        Stats<TF>& stats, Cross<TF>& cross, Dump<TF>& dump, Column<TF>& column)
{
    const std::string group_name = "thermo";

    // Add variables to the statistics
    if (stats.get_switch())
    {
        // Time series
        stats.add_time_series("rr", "Mean surface rain rate", "kg m-2 s-1", group_name);
        stats.add_time_series("rs", "Mean surface snow rate", "kg m-2 s-1", group_name);
        stats.add_time_series("rg", "Mean surface graupel rate", "kg m-2 s-1", group_name);

        stats.add_tendency(*fields.st.at("thl"), "z", tend_name, tend_longname);
        stats.add_tendency(*fields.st.at("qt") , "z", tend_name, tend_longname);
        stats.add_tendency(*fields.st.at("qr") , "z", tend_name, tend_longname);
        stats.add_tendency(*fields.st.at("qs") , "z", tend_name, tend_longname);
        stats.add_tendency(*fields.st.at("qg") , "z", tend_name, tend_longname);
    }

    if (column.get_switch())
    {
        column.add_time_series("rr", "Surface rain rate", "kg m-2 s-1");
        column.add_time_series("rs", "Surface snow rate", "kg m-2 s-1");
        column.add_time_series("rg", "Surface graupel rate", "kg m-2 s-1");
    }

    // Create cross sections
    // 1. Variables that this class can calculate/provide:
    const std::vector<std::string> allowed_crossvars = {"rr_bot", "rs_bot", "rg_bot", "rrsg_bot"};

    // 2. Cross-reference with the variables requested in the .ini file:
    crosslist = cross.get_enabled_variables(allowed_crossvars);

    // Create dumps
    // 1. Variables that this class can calculate/provide:
    const std::vector<std::string> allowed_dumpvars = {"qrsg_tend_sed", "qr_tend_sed", "qsg_tend_sed", "qtr_tend_frz", "thl_tend_micro", "qt_tend_micro"};

    // 2. Cross-reference with the variables requested in the .ini file:
    std::vector<std::string>& dumplist_global = dump.get_dumplist();
    auto it = dumplist_global.begin();
    while (it != dumplist_global.end())
    {
        if (std::count(allowed_dumpvars.begin(), allowed_dumpvars.end(), *it))
        {
            dumplist.push_back(*it);
            it = dumplist_global.erase(it);
        }
        else
            ++it;
    }

    if (std::count(dumplist.begin(), dumplist.end(), "qtr_tend_frz"))
    {
        auto& gd = grid.get_grid_data();
        fields.init_diagnostic_field(
                "qtr_tend_frz", "Net freezing rate of qt and qr into qs/qg", "kg kg-1 s-1", group_name, gd.sloc);
        fields.sd.at("qtr_tend_frz")->init();
    }

    if (std::count(dumplist.begin(), dumplist.end(), "thl_tend_micro"))
    {
        auto& gd = grid.get_grid_data();
        fields.init_diagnostic_field(
                "thl_tend_micro", "Microphysics tendency of thl", "K s-1", group_name, gd.sloc);
        fields.sd.at("thl_tend_micro")->init();
    }

    if (std::count(dumplist.begin(), dumplist.end(), "qt_tend_micro"))
    {
        auto& gd = grid.get_grid_data();
        fields.init_diagnostic_field(
                "qt_tend_micro", "Microphysics tendency of qt", "kg kg-1 s-1", group_name, gd.sloc);
        fields.sd.at("qt_tend_micro")->init();
    }
}

#ifndef USECUDA
template<typename TF>
void Microphys_nsw6<TF>::exec(Thermo<TF>& thermo, const double dt, Stats<TF>& stats)
{
    auto& gd = grid.get_grid_data();

    // Get liquid water, ice and pressure variables before starting.
    auto ql = fields.get_tmp();
    auto qi = fields.get_tmp();

    thermo.get_ql_qi(ql->fld, qi->fld);

    const std::vector<TF>& p = thermo.get_basestate_vector("p");
    const std::vector<TF>& exner = thermo.get_basestate_vector("exner");

    const auto net_frz_it = fields.sd.find("qtr_tend_frz");
    TF* const net_frz = (net_frz_it != fields.sd.end()) ? net_frz_it->second->fld.data() : nullptr;

    if (net_frz != nullptr)
        std::fill(net_frz_it->second->fld.begin(), net_frz_it->second->fld.end(), TF(0.));

    const auto thl_tend_micro_it = fields.sd.find("thl_tend_micro");
    TF* const thl_tend_micro = (thl_tend_micro_it != fields.sd.end()) ? thl_tend_micro_it->second->fld.data() : nullptr;

    if (thl_tend_micro != nullptr)
        std::fill(thl_tend_micro_it->second->fld.begin(), thl_tend_micro_it->second->fld.end(), TF(0.));

    const auto qt_tend_micro_it = fields.sd.find("qt_tend_micro");
    TF* const qt_tend_micro = (qt_tend_micro_it != fields.sd.end()) ? qt_tend_micro_it->second->fld.data() : nullptr;

    if (qt_tend_micro != nullptr)
        std::fill(qt_tend_micro_it->second->fld.begin(), qt_tend_micro_it->second->fld.end(), TF(0.));

    conversion(
            fields.st.at("qr")->fld.data(), fields.st.at("qs")->fld.data(), fields.st.at("qg")->fld.data(),
            fields.st.at("qt")->fld.data(), fields.st.at("thl")->fld.data(),
            net_frz,
            thl_tend_micro,
            qt_tend_micro,
            fields.sp.at("qr")->fld.data(), fields.sp.at("qs")->fld.data(), fields.sp.at("qg")->fld.data(),
            fields.sp.at("qt")->fld.data(), fields.sp.at("thl")->fld.data(),
            ql->fld.data(), qi->fld.data(),
            fields.rhoref.data(), exner.data(), p.data(),
            gd.dzi.data(), gd.dzhi.data(),
            this->Nc0, TF(dt),
            gd.istart, gd.jstart, gd.kstart,
            gd.iend, gd.jend, gd.kend,
            gd.icells, gd.ijcells);

    fields.release_tmp(ql);
    fields.release_tmp(qi);

    auto tmp1 = fields.get_tmp();
    auto tmp2 = fields.get_tmp();
    auto tmp3 = fields.get_tmp();
    auto tmp4 = fields.get_tmp();
    auto tmp_xy1 = fields.get_tmp_xy();
    auto tmp_xy2 = fields.get_tmp_xy();
    auto tmp_xy3 = fields.get_tmp_xy();

    // Falling rain.
    sedimentation_ss08(
            fields.st.at("qr")->fld.data(), rr_bot.data(),
            tmp1->fld.data(), tmp2->fld.data(),
            tmp3->fld.data(), tmp4->fld.data(),
            tmp_xy1->data(), tmp_xy2->data(), tmp_xy3->data(),
            fields.sp.at("qr")->fld.data(),
            fields.rhoref.data(),
            gd.dzi.data(), gd.dz.data(),
            dt,
            a_r<TF>, b_r<TF>, c_r<TF>, d_r<TF>, N_0r<TF>,
            qr_min<TF>,
            gd.istart, gd.jstart, gd.kstart,
            gd.iend, gd.jend, gd.kend,
            gd.icells, gd.ijcells);

    // Falling snow.
    sedimentation_ss08(
            fields.st.at("qs")->fld.data(), rs_bot.data(),
            tmp1->fld.data(), tmp2->fld.data(),
            tmp3->fld.data(), tmp4->fld.data(),
            tmp_xy1->data(), tmp_xy2->data(), tmp_xy3->data(),
            fields.sp.at("qs")->fld.data(),
            fields.rhoref.data(),
            gd.dzi.data(), gd.dz.data(),
            dt,
            a_s<TF>, b_s<TF>, c_s<TF>, d_s<TF>, N_0s<TF>,
            qs_min<TF>,
            gd.istart, gd.jstart, gd.kstart,
            gd.iend, gd.jend, gd.kend,
            gd.icells, gd.ijcells);

    // Falling graupel.
    sedimentation_ss08(
            fields.st.at("qg")->fld.data(), rg_bot.data(),
            tmp1->fld.data(), tmp2->fld.data(),
            tmp3->fld.data(), tmp4->fld.data(),
            tmp_xy1->data(), tmp_xy2->data(), tmp_xy3->data(),
            fields.sp.at("qg")->fld.data(),
            fields.rhoref.data(),
            gd.dzi.data(), gd.dz.data(),
            dt,
            a_g<TF>, b_g<TF>, c_g<TF>, d_g<TF>, N_0g<TF>,
            qg_min<TF>,
            gd.istart, gd.jstart, gd.kstart,
            gd.iend, gd.jend, gd.kend,
            gd.icells, gd.ijcells);

    fields.release_tmp(tmp1);
    fields.release_tmp(tmp2);
    fields.release_tmp(tmp3);
    fields.release_tmp(tmp4);
    fields.release_tmp_xy(tmp_xy1);
    fields.release_tmp_xy(tmp_xy2);
    fields.release_tmp_xy(tmp_xy3);

    stats.calc_tend(*fields.st.at("thl"), tend_name);
    stats.calc_tend(*fields.st.at("qt" ), tend_name);
    stats.calc_tend(*fields.st.at("qr" ), tend_name);
    stats.calc_tend(*fields.st.at("qs" ), tend_name);
    stats.calc_tend(*fields.st.at("qg" ), tend_name);
}
#endif

template<typename TF>
void Microphys_nsw6<TF>::exec_stats(Stats<TF>& stats, Thermo<TF>& thermo, const double dt)
{
    // Time series
    const TF no_offset = 0.;
    stats.calc_stats_2d("rr", rr_bot, no_offset);
    stats.calc_stats_2d("rs", rs_bot, no_offset);
    stats.calc_stats_2d("rg", rg_bot, no_offset);
}

#ifndef USECUDA
template<typename TF>
void Microphys_nsw6<TF>::exec_column(Column<TF>& column)
{
    const TF no_offset = 0.;
    column.calc_time_series("rr", rr_bot.data(), no_offset);
    column.calc_time_series("rs", rs_bot.data(), no_offset);
    column.calc_time_series("rg", rg_bot.data(), no_offset);
}
#endif

template<typename TF>
void Microphys_nsw6<TF>::exec_cross(Cross<TF>& cross, unsigned long iotime)
{
    TF no_offset = 0.;
    if (cross.get_switch())
    {
        for (auto& it : crosslist)
        {
            if (it == "rr_bot")
                cross.cross_plane(rr_bot.data(), no_offset, "rr_bot", iotime);

            if (it == "rs_bot")
                cross.cross_plane(rs_bot.data(), no_offset, "rs_bot", iotime);

            if (it == "rg_bot")
                cross.cross_plane(rg_bot.data(), no_offset, "rg_bot", iotime);

            if (it == "rrsg_bot")
            {
                auto rrsg_bot = fields.get_tmp_xy();
                calc_rrsg_bot(rrsg_bot->data(), rr_bot.data(), rs_bot.data(), rg_bot.data(), rr_bot.size());
                cross.cross_plane(rrsg_bot->data(), no_offset, "rrsg_bot", iotime);
                fields.release_tmp_xy(rrsg_bot);
            }
        }
    }
}

template<typename TF>
void Microphys_nsw6<TF>::exec_dump(Dump<TF>& dump, unsigned long iotime, const double dt)
{
    auto& gd = grid.get_grid_data();

    for (auto& it : dumplist)
    {
        if (it == "qrsg_tend_sed" || it == "qr_tend_sed" || it == "qsg_tend_sed")
        {
            auto qpt = fields.get_tmp();
            auto rc_bot_tmp = fields.get_tmp_xy();
            auto w_qc = fields.get_tmp();
            auto c_qc = fields.get_tmp();
            auto slope_qc = fields.get_tmp();
            auto flux_qc = fields.get_tmp();
            auto ftot_v = fields.get_tmp_xy();
            auto dzz_v = fields.get_tmp_xy();
            auto cc_v = fields.get_tmp_xy();

            std::fill(qpt->fld.begin(), qpt->fld.end(), TF(0.));

            auto add_sedimentation_tendency = [&](
                    const std::string& varname,
                    const TF a, const TF b, const TF c, const TF d, const TF N_0, const TF q_min)
            {
                sedimentation_ss08(
                        qpt->fld.data(), rc_bot_tmp->data(),
                        w_qc->fld.data(), c_qc->fld.data(),
                        slope_qc->fld.data(), flux_qc->fld.data(),
                        ftot_v->data(), dzz_v->data(), cc_v->data(),
                        fields.sp.at(varname)->fld.data(),
                        fields.rhoref.data(),
                        gd.dzi.data(), gd.dz.data(),
                        dt,
                        a, b, c, d, N_0,
                        q_min,
                        gd.istart, gd.jstart, gd.kstart,
                        gd.iend, gd.jend, gd.kend,
                        gd.icells, gd.ijcells);
            };

            // Warm (rain) and cold (snow + graupel) precipitation, or their sum.
            if (it != "qsg_tend_sed")
                add_sedimentation_tendency("qr", a_r<TF>, b_r<TF>, c_r<TF>, d_r<TF>, N_0r<TF>, qr_min<TF>);

            if (it != "qr_tend_sed")
            {
                add_sedimentation_tendency("qs", a_s<TF>, b_s<TF>, c_s<TF>, d_s<TF>, N_0s<TF>, qs_min<TF>);
                add_sedimentation_tendency("qg", a_g<TF>, b_g<TF>, c_g<TF>, d_g<TF>, N_0g<TF>, qg_min<TF>);
            }

            dump.save_dump(qpt->fld.data(), it, iotime);

            fields.release_tmp(qpt);
            fields.release_tmp_xy(rc_bot_tmp);
            fields.release_tmp(w_qc);
            fields.release_tmp(c_qc);
            fields.release_tmp(slope_qc);
            fields.release_tmp(flux_qc);
            fields.release_tmp_xy(ftot_v);
            fields.release_tmp_xy(dzz_v);
            fields.release_tmp_xy(cc_v);
        }

        if (it == "qtr_tend_frz")
            dump.save_dump(fields.sd.at("qtr_tend_frz")->fld.data(), "qtr_tend_frz", iotime);

        if (it == "thl_tend_micro")
            dump.save_dump(fields.sd.at("thl_tend_micro")->fld.data(), "thl_tend_micro", iotime);

        if (it == "qt_tend_micro")
            dump.save_dump(fields.sd.at("qt_tend_micro")->fld.data(), "qt_tend_micro", iotime);
    }
}

#ifndef USECUDA
template<typename TF>
unsigned long Microphys_nsw6<TF>::get_time_limit(unsigned long idt, const double dt)
{
    auto& gd = grid.get_grid_data();

    auto tmp = fields.get_tmp();

    double cfl = 0.;

    // Compute CFL for rain.
    const double cfl_r = calc_cfl_ss08(
            tmp->fld.data(),
            fields.sp.at("qr")->fld.data(),
            fields.rhoref.data(),
            gd.dzi.data(), gd.dz.data(),
            dt,
            a_r<TF>, b_r<TF>, c_r<TF>, d_r<TF>, N_0r<TF>,
            qr_min<TF>,
            gd.istart, gd.jstart, gd.kstart,
            gd.iend, gd.jend, gd.kend,
            gd.icells, gd.ijcells);
    cfl = cfl_r;

    // Compute CFL for snow.
    const double cfl_s = calc_cfl_ss08(
            tmp->fld.data(),
            fields.sp.at("qs")->fld.data(),
            fields.rhoref.data(),
            gd.dzi.data(), gd.dz.data(),
            dt,
            a_s<TF>, b_s<TF>, c_s<TF>, d_s<TF>, N_0s<TF>,
            qs_min<TF>,
            gd.istart, gd.jstart, gd.kstart,
            gd.iend, gd.jend, gd.kend,
            gd.icells, gd.ijcells);
    cfl = std::max(cfl, cfl_s);

    // Compute CFL for graupel.
    const double cfl_g = calc_cfl_ss08(
            tmp->fld.data(),
            fields.sp.at("qg")->fld.data(),
            fields.rhoref.data(),
            gd.dzi.data(), gd.dz.data(),
            dt,
            a_g<TF>, b_g<TF>, c_g<TF>, d_g<TF>, N_0g<TF>,
            qg_min<TF>,
            gd.istart, gd.jstart, gd.kstart,
            gd.iend, gd.jend, gd.kend,
            gd.icells, gd.ijcells);
    cfl = std::max(cfl, cfl_g);

    // Get maximum CFL across all MPI tasks
    master.max(&cfl, 1);
    fields.release_tmp(tmp);

    // Prevent zero division.
    cfl = std::max(cfl, 1.e-5);

    return idt * this->cflmax / cfl;
}
#endif

template<typename TF>
bool Microphys_nsw6<TF>::has_mask(std::string name)
{
    return false;
}

template<typename TF>
void Microphys_nsw6<TF>::get_mask(Stats<TF>& stats, std::string mask_name)
{
    std::string message = "NSW6 microphysics scheme can not provide mask: \"" + mask_name +"\"";
    throw std::runtime_error(message);
}

template<typename TF>
void Microphys_nsw6<TF>::get_surface_rain_rate(std::vector<TF>& field)
{
    // Make a hard copy of the surface rain precipitation field
    field = rr_bot;

    // Add snow and graupel surface precipitation
    std::transform(field.begin(), field.end(), rs_bot.begin(), field.begin(), std::plus<TF>());
    std::transform(field.begin(), field.end(), rg_bot.begin(), field.begin(), std::plus<TF>());
}


#ifdef FLOAT_SINGLE
template class Microphys_nsw6<float>;
#else
template class Microphys_nsw6<double>;
#endif
