/*
 * MicroHH
 * Copyright (c) 2011-2015 Chiel van Heerwaarden
 * Copyright (c) 2011-2015 Thijs Heus
 * Copyright (c) 2014-2015 Bart van Stratum
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

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "master.h"
#include "input.h"
#include "grid.h"
#include "fields.h"
#include "boundary_surface.h"
#include "defines.h"
#include "constants.h"
#include "thermo.h"
#include "model.h"
#include "master.h"
#include "cross.h"
#include "monin_obukhov.h"

namespace
{
    // Make a shortcut in the file scope.
    namespace most = Monin_obukhov;
    // Size of the lookup table.
    const int nzL = 10000; // Size of the lookup table for MO iterations.
}

Boundary_surface::Boundary_surface(Model* modelin, Input* inputin) : Boundary(modelin, inputin)
{
    swboundary = "surface";

    ustar = 0;
    obuk  = 0;
    nobuk = 0;
    zL_sl = 0;
    f_sl  = 0;

#ifdef USECUDA
    ustar_g = 0;
    obuk_g  = 0;
    nobuk_g = 0;
    zL_sl_g = 0;
    f_sl_g  = 0;
#endif
}

Boundary_surface::~Boundary_surface()
{
    delete[] ustar;
    delete[] obuk;
    delete[] nobuk;
    delete[] zL_sl;
    delete[] f_sl;

#ifdef USECUDA
    clear_device();
#endif
}

void Boundary_surface::create(Input *inputin)
{
    process_time_dependent(inputin);

    // add variables to the statistics
    if (stats->get_switch() == "1")
    {
        stats->add_time_series("ustar", "Surface friction velocity", "m s-1");
        stats->add_time_series("obuk", "Obukhov length", "m");
    }

    // Prepare the lookup table for the surface solver
    init_solver();

    // in case the momentum has a fixed ustar, set the value to that of the input
    if (mbcbot == Ustar_type)
        set_ustar();
}

void Boundary_surface::init(Input *inputin)
{
    // 1. Process the boundary conditions now all fields are registered
    process_bcs(inputin);

    // 2. Read and check the boundary_surface specific settings
    process_input(inputin);

    // 3. Allocate and initialize the 2D surface fields
    init_surface();
}

void Boundary_surface::process_input(Input *inputin)
{
    int nerror = 0;

    nerror += inputin->get_item(&z0m, "boundary", "z0m", "");
    nerror += inputin->get_item(&z0h, "boundary", "z0h", "");

    // crash in case fixed gradient is prescribed
    if (mbcbot == Neumann_type)
    {
        master->print_error("Neumann bc is not supported in surface model\n");
        ++nerror;
    }
    // read the ustar value only if fixed fluxes are prescribed
    else if (mbcbot == Ustar_type)
        nerror += inputin->get_item(&ustarin, "boundary", "ustar", "");

    // process the scalars
    for (BcMap::const_iterator it=sbc.begin(); it!=sbc.end(); ++it)
    {
        // crash in case fixed gradient is prescribed
        if (it->second->bcbot == Neumann_type)
        {
            master->print_error("fixed gradient bc is not supported in surface model\n");
            ++nerror;
        }

        // crash in case of fixed momentum flux and dirichlet bc for scalar
        if (it->second->bcbot == Dirichlet_type && mbcbot == Ustar_type)
        {
            master->print_error("fixed Ustar bc in combination with Dirichlet bc for scalars is not supported\n");
            ++nerror;
        }
    }

    // check whether the prognostic thermo vars are of the same type
    std::vector<std::string> thermolist;
    model->thermo->get_prog_vars(&thermolist);

    std::vector<std::string>::const_iterator it = thermolist.begin();

    // save the bc of the first thermo field in case thermo is enabled
    if (it != thermolist.end())
        thermobc = sbc[*it]->bcbot;

    while (it != thermolist.end())
    {
        if (sbc[*it]->bcbot != thermobc)
        {
            ++nerror;
            master->print_error("all thermo variables need to have the same bc type\n");
        }
        ++it;
    }

    if (nerror)
        throw 1;

    // Cross sections
    allowedcrossvars.push_back("ustar");
    allowedcrossvars.push_back("obuk");

    // Read list of cross sections
    nerror += inputin->get_list(&crosslist , "boundary", "crosslist" , "");

    // Get global cross-list from cross.cxx
    std::vector<std::string> *crosslist_global = model->cross->get_crosslist(); 

    // Check input list of cross variables (crosslist)
    std::vector<std::string>::iterator it2=crosslist_global->begin();
    while (it2 != crosslist_global->end())
    {
        if (std::count(allowedcrossvars.begin(),allowedcrossvars.end(),*it2))
        {
            // Remove variable from global list, put in local list
            crosslist.push_back(*it2);
            crosslist_global->erase(it2); // erase() returns iterator of next element..
        }
        else
            ++it2;
    }
}

void Boundary_surface::init_surface()
{
    obuk  = new double[grid->ijcells];
    nobuk = new int   [grid->ijcells];
    ustar = new double[grid->ijcells];

    stats = model->stats;

    const int jj = grid->icells;

    // initialize the obukhov length on a small number
    for (int j=0; j<grid->jcells; ++j)
        #pragma ivdep
        for (int i=0; i<grid->icells; ++i)
        {
            const int ij = i + j*jj;
            obuk[ij]  = Constants::dsmall;
            nobuk[ij] = 0;
        }


}

void Boundary_surface::exec_cross()
{
    int nerror = 0;

    for (std::vector<std::string>::const_iterator it=crosslist.begin(); it<crosslist.end(); ++it)
    {
        if (*it == "ustar")
            nerror += model->cross->cross_plane(ustar, fields->atmp["tmp1"]->data, "ustar");
        else if (*it == "obuk")
            nerror += model->cross->cross_plane(obuk,  fields->atmp["tmp1"]->data, "obuk");
    }  

    if (nerror)
        throw 1;
}

void Boundary_surface::exec_stats(Mask *m)
{
    stats->calc_mean2d(&m->tseries["obuk"].data , obuk , 0., fields->atmp["tmp4"]->databot, &stats->nmaskbot);
    stats->calc_mean2d(&m->tseries["ustar"].data, ustar, 0., fields->atmp["tmp4"]->databot, &stats->nmaskbot);
}

void Boundary_surface::set_ustar()
{
    const int jj = grid->icells;

    set_bc(fields->u->databot, fields->u->datagradbot, fields->u->datafluxbot, Dirichlet_type, ubot, fields->visc, grid->utrans);
    set_bc(fields->v->databot, fields->v->datagradbot, fields->v->datafluxbot, Dirichlet_type, vbot, fields->visc, grid->vtrans);

    for (int j=0; j<grid->jcells; ++j)
            #pragma ivdep
            for (int i=0; i<grid->icells; ++i)
            {
                const int ij = i + j*jj;
                // Limit ustar at 1e-4 to avoid zero divisions.
                ustar[ij] = std::max(0.0001, ustarin);
            }
}

// Prepare the surface layer solver.
void Boundary_surface::init_solver()
{
    zL_sl = new float[nzL];
    f_sl  = new float[nzL];

    double* zL_tmp = new double[nzL];

    // Calculate the non-streched part between -5 to 10 z/L with 9/10 of the points,
    // and stretch up to -1e4 in the negative limit.
    // Alter next three values in case the range need to be changed.
    const double zL_min = -1.e4;
    const double zLrange_min = -5.;
    const double zLrange_max = 10.;

    double dzL = (zLrange_max - zLrange_min) / (9.*nzL/10.-1.);
    zL_tmp[0] = -zLrange_max;
    for (int n=1; n<9*nzL/10; ++n)
        zL_tmp[n] = zL_tmp[n-1] + dzL;

    // Stretch the remainder of the z/L values far down for free convection.
    const double zLend = -(zL_min - zLrange_min);

    // Find stretching that ends up at the correct value using geometric progression.
    double r  = 1.01;
    double r0 = Constants::dhuge;
    while (std::abs( (r-r0)/r0 ) > 1.e-10)
    {
        r0 = r;
        r  = std::pow( 1. - (zLend/dzL)*(1.-r), (1./ (nzL/10.) ) );
    }

    for (int n=9*nzL/10; n<nzL; ++n)
    {
        zL_tmp[n] = zL_tmp[n-1] + dzL;
        dzL *= r;
    }

    // Calculate the final array and delete the temporary array.
    for (int n=0; n<nzL; ++n)
        zL_sl[n] = -zL_tmp[nzL-n-1];

    delete[] zL_tmp;

    // Calculate the evaluation function.
    if (mbcbot == Dirichlet_type && thermobc == Flux_type)
    {
        const double zsl = grid->z[grid->kstart];
        for (int n=0; n<nzL; ++n)
            f_sl[n] = zL_sl[n] * std::pow(most::fm(zsl, z0m, zsl/zL_sl[n]), 3);
    }
    else if (mbcbot == Dirichlet_type && thermobc == Dirichlet_type)
    {
        const double zsl = grid->z[grid->kstart];
        for (int n=0; n<nzL; ++n)
            f_sl[n] = zL_sl[n] * std::pow(most::fm(zsl, z0m, zsl/zL_sl[n]), 2) / most::fh(zsl, z0h, zsl/zL_sl[n]);
    }
}

#ifndef USECUDA
void Boundary_surface::update_bcs()
{
    // Start with retrieving the stability information.
    if (model->thermo->get_switch() == "0")
    {
        stability_neutral(ustar, obuk,
                          fields->u->data, fields->v->data,
                          fields->u->databot, fields->v->databot,
                          fields->atmp["tmp1"]->data, grid->z);
    }
    else
    {
        // Store the buoyancy in tmp1.
        model->thermo->get_buoyancy_surf(fields->atmp["tmp1"]);
        stability(ustar, obuk, fields->atmp["tmp1"]->datafluxbot,
                  fields->u->data,    fields->v->data,    fields->atmp["tmp1"]->data,
                  fields->u->databot, fields->v->databot, fields->atmp["tmp1"]->databot,
                  fields->atmp["tmp2"]->data, grid->z);
    }

    // Calculate the surface value, gradient and flux depending on the chosen boundary condition.
    surfm(ustar, obuk,
          fields->u->data, fields->u->databot, fields->u->datagradbot, fields->u->datafluxbot,
          fields->v->data, fields->v->databot, fields->v->datagradbot, fields->v->datafluxbot,
          grid->z[grid->kstart], mbcbot);

    for (FieldMap::const_iterator it=fields->sp.begin(); it!=fields->sp.end(); ++it)
    {
        surfs(ustar, obuk, it->second->data,
              it->second->databot, it->second->datagradbot, it->second->datafluxbot,
              grid->z[grid->kstart], sbc[it->first]->bcbot);
    }
}
#endif

void Boundary_surface::update_slave_bcs()
{
    // This function does nothing when the surface model is enabled, because 
    // the fields are computed by the surface model in update_bcs.
}

void Boundary_surface::stability(double* restrict ustar, double* restrict obuk, double* restrict bfluxbot,
                                 double* restrict u    , double* restrict v   , double* restrict b       ,
                                 double* restrict ubot , double* restrict vbot, double* restrict bbot    ,
                                 double* restrict dutot, double* restrict z)
{
    const int ii = 1;
    const int jj = grid->icells;
    const int kk = grid->ijcells;

    const int kstart = grid->kstart;

    // calculate total wind
    double du2;
    //double utot, ubottot, du2;
    const double minval = 1.e-1;
    // first, interpolate the wind to the scalar location
    for (int j=grid->jstart; j<grid->jend; ++j)
#pragma ivdep
        for (int i=grid->istart; i<grid->iend; ++i)
        {
            const int ij  = i + j*jj;
            const int ijk = i + j*jj + kstart*kk;
            du2 = std::pow(0.5*(u[ijk] + u[ijk+ii]) - 0.5*(ubot[ij] + ubot[ij+ii]), 2)
                + std::pow(0.5*(v[ijk] + v[ijk+jj]) - 0.5*(vbot[ij] + vbot[ij+jj]), 2);
            // prevent the absolute wind gradient from reaching values less than 0.01 m/s,
            // otherwise evisc at k = kstart blows up
            dutot[ij] = std::max(std::pow(du2, 0.5), minval);
        }

    grid->boundary_cyclic_2d(dutot);

    // calculate Obukhov length
    // case 1: fixed buoyancy flux and fixed ustar
    if (mbcbot == Ustar_type && thermobc == Flux_type)
    {
        for (int j=0; j<grid->jcells; ++j)
#pragma ivdep
            for (int i=0; i<grid->icells; ++i)
            {
                const int ij = i + j*jj;
                obuk[ij] = -std::pow(ustar[ij], 3) / (Constants::kappa*bfluxbot[ij]);
            }
    }
    // case 2: fixed buoyancy surface value and free ustar
    else if (mbcbot == Dirichlet_type && thermobc == Flux_type)
    {
        for (int j=0; j<grid->jcells; ++j)
#pragma ivdep
            for (int i=0; i<grid->icells; ++i)
            {
                const int ij = i + j*jj;
                obuk [ij] = calc_obuk_noslip_flux(zL_sl, f_sl, nobuk[ij], dutot[ij], bfluxbot[ij], z[kstart]);
                ustar[ij] = dutot[ij] * most::fm(z[kstart], z0m, obuk[ij]);
            }
    }
    else if (mbcbot == Dirichlet_type && thermobc == Dirichlet_type)
    {
        for (int j=0; j<grid->jcells; ++j)
#pragma ivdep
            for (int i=0; i<grid->icells; ++i)
            {
                const int ij  = i + j*jj;
                const int ijk = i + j*jj + kstart*kk;
                const double db = b[ijk] - bbot[ij];
                obuk [ij] = calc_obuk_noslip_dirichlet(zL_sl, f_sl, nobuk[ij], dutot[ij], db, z[kstart]);
                ustar[ij] = dutot[ij] * most::fm(z[kstart], z0m, obuk[ij]);
            }
    }
}

void Boundary_surface::stability_neutral(double* restrict ustar, double* restrict obuk,
                                         double* restrict u    , double* restrict v   ,
                                         double* restrict ubot , double* restrict vbot,
                                         double* restrict dutot, double* restrict z)
{
    const int ii = 1;
    const int jj = grid->icells;
    const int kk = grid->ijcells;

    const int kstart = grid->kstart;

    // calculate total wind
    double du2;
    const double minval = 1.e-1;

    // first, interpolate the wind to the scalar location
    for (int j=grid->jstart; j<grid->jend; ++j)
        #pragma ivdep
        for (int i=grid->istart; i<grid->iend; ++i)
        {
            const int ij  = i + j*jj;
            const int ijk = i + j*jj + kstart*kk;
            du2 = std::pow(0.5*(u[ijk] + u[ijk+ii]) - 0.5*(ubot[ij] + ubot[ij+ii]), 2)
                + std::pow(0.5*(v[ijk] + v[ijk+jj]) - 0.5*(vbot[ij] + vbot[ij+jj]), 2);
            // prevent the absolute wind gradient from reaching values less than 0.01 m/s,
            // otherwise evisc at k = kstart blows up
            dutot[ij] = std::max(std::pow(du2, 0.5), minval);
        }

    grid->boundary_cyclic_2d(dutot);

    // Set the Obukhov length to a very large negative number
    // Case 1: fixed ustar, only set the Obukhov length
    if (mbcbot == Ustar_type)
    {
        for (int j=grid->jstart; j<grid->jend; ++j)
            #pragma ivdep
            for (int i=grid->istart; i<grid->iend; ++i)
            {
                const int ij = i + j*jj;
                obuk[ij] = -Constants::dbig;
            }
    }
    // Case 2: free ustar, calculate Obukhov length
    else
    {
        for (int j=0; j<grid->jcells; ++j)
            #pragma ivdep
            for (int i=0; i<grid->icells; ++i)
            {
                const int ij = i + j*jj;
                obuk [ij] = -Constants::dbig;
                ustar[ij] = dutot[ij] * most::fm(z[kstart], z0m, obuk[ij]);
            }
    }
}

void Boundary_surface::surfm(double* restrict ustar, double* restrict obuk, 
                             double* restrict u, double* restrict ubot, double* restrict ugradbot, double* restrict ufluxbot, 
                             double* restrict v, double* restrict vbot, double* restrict vgradbot, double* restrict vfluxbot, 
                             double zsl, int bcbot)
{
    const int ii = 1;
    const int jj = grid->icells;
    const int kk = grid->ijcells;

    const int kstart = grid->kstart;

    // the surface value is known, calculate the flux and gradient
    if (bcbot == Dirichlet_type)
    {
        // first calculate the surface value
        for (int j=grid->jstart; j<grid->jend; ++j)
#pragma ivdep
            for (int i=grid->istart; i<grid->iend; ++i)
            {
                const int ij  = i + j*jj;
                const int ijk = i + j*jj + kstart*kk;

                // interpolate the whole stability function rather than ustar or obuk
                ufluxbot[ij] = -(u[ijk]-ubot[ij])*0.5*(ustar[ij-ii]*most::fm(zsl, z0m, obuk[ij-ii]) + ustar[ij]*most::fm(zsl, z0m, obuk[ij]));
                vfluxbot[ij] = -(v[ijk]-vbot[ij])*0.5*(ustar[ij-jj]*most::fm(zsl, z0m, obuk[ij-jj]) + ustar[ij]*most::fm(zsl, z0m, obuk[ij]));
            }

        grid->boundary_cyclic_2d(ufluxbot);
        grid->boundary_cyclic_2d(vfluxbot);
    }
    // the flux is known, calculate the surface value and gradient
    else if (bcbot == Ustar_type)
    {
        // first redistribute ustar over the two flux components
        double u2,v2,vonu2,uonv2,ustaronu4,ustaronv4;
        const double minval = 1.e-2;

        for (int j=grid->jstart; j<grid->jend; ++j)
#pragma ivdep
            for (int i=grid->istart; i<grid->iend; ++i)
            {
                const int ij  = i + j*jj;
                const int ijk = i + j*jj + kstart*kk;
                // minimize the wind at 0.01, thus the wind speed squared at 0.0001
                vonu2 = std::max(minval, 0.25*( std::pow(v[ijk-ii]-vbot[ij-ii], 2) + std::pow(v[ijk-ii+jj]-vbot[ij-ii+jj], 2)
                            + std::pow(v[ijk   ]-vbot[ij   ], 2) + std::pow(v[ijk   +jj]-vbot[ij   +jj], 2)) );
                uonv2 = std::max(minval, 0.25*( std::pow(u[ijk-jj]-ubot[ij-jj], 2) + std::pow(u[ijk+ii-jj]-ubot[ij+ii-jj], 2)
                            + std::pow(u[ijk   ]-ubot[ij   ], 2) + std::pow(u[ijk+ii   ]-ubot[ij+ii   ], 2)) );
                u2 = std::max(minval, std::pow(u[ijk]-ubot[ij], 2) );
                v2 = std::max(minval, std::pow(v[ijk]-vbot[ij], 2) );
                ustaronu4 = 0.5*(std::pow(ustar[ij-ii], 4) + std::pow(ustar[ij], 4));
                ustaronv4 = 0.5*(std::pow(ustar[ij-jj], 4) + std::pow(ustar[ij], 4));
                ufluxbot[ij] = -copysign(1., u[ijk]-ubot[ij]) * std::pow(ustaronu4 / (1. + vonu2 / u2), 0.5);
                vfluxbot[ij] = -copysign(1., v[ijk]-vbot[ij]) * std::pow(ustaronv4 / (1. + uonv2 / v2), 0.5);
            }

        grid->boundary_cyclic_2d(ufluxbot);
        grid->boundary_cyclic_2d(vfluxbot);

        // CvH: I think that the problem is not closed, since both the fluxes and the surface values
        // of u and v are unknown. You have to assume a no slip in order to get the fluxes and therefore
        // should not update the surface values with those that belong to the flux. This procedure needs
        // to be checked more carefully.
        /*
        // calculate the surface values
        for (int j=grid->jstart; j<grid->jend; ++j)
#pragma ivdep
for (int i=grid->istart; i<grid->iend; ++i)
{
ij  = i + j*jj;
ijk = i + j*jj + kstart*kk;
        // interpolate the whole stability function rather than ustar or obuk
        ubot[ij] = 0.;// ufluxbot[ij] / (0.5*(ustar[ij-ii]*fm(zsl, z0m, obuk[ij-ii]) + ustar[ij]*fm(zsl, z0m, obuk[ij]))) + u[ijk];
        vbot[ij] = 0.;// vfluxbot[ij] / (0.5*(ustar[ij-jj]*fm(zsl, z0m, obuk[ij-jj]) + ustar[ij]*fm(zsl, z0m, obuk[ij]))) + v[ijk];
        }

        grid->boundary_cyclic_2d(ubot);
        grid->boundary_cyclic_2d(vbot);
        */
    }

    for (int j=0; j<grid->jcells; ++j)
#pragma ivdep
        for (int i=0; i<grid->icells; ++i)
        {
            const int ij  = i + j*jj;
            const int ijk = i + j*jj + kstart*kk;
            // use the linearly interpolated grad, rather than the MO grad,
            // to prevent giving unresolvable gradients to advection schemes
            // vargradbot[ij] = -varfluxbot[ij] / (kappa*z0m*ustar[ij]) * phih(zsl/obuk[ij]);
            ugradbot[ij] = (u[ijk]-ubot[ij])/zsl;
            vgradbot[ij] = (v[ijk]-vbot[ij])/zsl;
        }
}

void Boundary_surface::surfs(double* restrict ustar, double* restrict obuk, double* restrict var,
                             double* restrict varbot, double* restrict vargradbot, double* restrict varfluxbot, 
                             double zsl, int bcbot)
{
    const int jj = grid->icells;
    const int kk = grid->ijcells;

    const int kstart = grid->kstart;

    // the surface value is known, calculate the flux and gradient
    if (bcbot == Dirichlet_type)
    {
        for (int j=0; j<grid->jcells; ++j)
#pragma ivdep
            for (int i=0; i<grid->icells; ++i)
            {
                const int ij  = i + j*jj;
                const int ijk = i + j*jj + kstart*kk;
                varfluxbot[ij] = -(var[ijk]-varbot[ij])*ustar[ij]*most::fh(zsl, z0h, obuk[ij]);
                // vargradbot[ij] = -varfluxbot[ij] / (kappa*z0h*ustar[ij]) * phih(zsl/obuk[ij]);
                // use the linearly interpolated grad, rather than the MO grad,
                // to prevent giving unresolvable gradients to advection schemes
                vargradbot[ij] = (var[ijk]-varbot[ij])/zsl;
            }
    }
    else if (bcbot == Flux_type)
    {
        // the flux is known, calculate the surface value and gradient
        for (int j=0; j<grid->jcells; ++j)
#pragma ivdep
            for (int i=0; i<grid->icells; ++i)
            {
                const int ij  = i + j*jj;
                const int ijk = i + j*jj + kstart*kk;
                varbot[ij] = varfluxbot[ij] / (ustar[ij]*most::fh(zsl, z0h, obuk[ij])) + var[ijk];
                // vargradbot[ij] = -varfluxbot[ij] / (kappa*z0h*ustar[ij]) * phih(zsl/obuk[ij]);
                // use the linearly interpolated grad, rather than the MO grad,
                // to prevent giving unresolvable gradients to advection schemes
                vargradbot[ij] = (var[ijk]-varbot[ij])/zsl;
            }
    }
}

namespace
{
    double find_zL(const float* const restrict zL, const float* const restrict f,
                   int &n, const float Ri)
    {
        // Determine search direction.
        if ( (f[n]-Ri) > 0 )
            while ( (f[n-1]-Ri) > 0 && n > 0) { --n; }
        else
            while ( (f[n]-Ri) < 0 && n < (nzL-1) ) { ++n; }

        const double zL0 = (n == 0 || n == nzL-1) ? zL[n] : zL[n-1] + (Ri-f[n-1]) / (f[n]-f[n-1]) * (zL[n]-zL[n-1]);

        return zL0;
    }
}

double Boundary_surface::calc_obuk_noslip_flux(const float* const restrict zL, const float* const restrict f,
                                               int& n,
                                               const double du, const double bfluxbot, const double zsl)
{
    // Calculate the appropriate Richardson number and reduce precision.
    const float Ri = -Constants::kappa * bfluxbot * zsl / std::pow(du, 3);

    return zsl/find_zL(zL, f, n, Ri);
}

double Boundary_surface::calc_obuk_noslip_dirichlet(const float* const restrict zL, const float* const restrict f,
                                                    int& n,
                                                    const double du, const double db, const double zsl)
{
    // Calculate the appropriate Richardson number and reduce precision.
    const float Ri = Constants::kappa * db * zsl / std::pow(du, 2);

    return zsl/find_zL(zL, f, n, Ri);
}
