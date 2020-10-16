#include <AMReX_REAL.H>
#include <AMReX_BLFort.H>
#include <AMReX_SPACE.H>

#ifdef AMREX_USE_EB
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBMultiFabUtil.H>
#endif

#include <AMReX_MultiFabUtil.H>
#include <AMReX_MacProjector.H>

#include <incflo.H>

using namespace amrex;

//
// Computes the following decomposition:
// 
//    u + c*grad(phi)/ro = u*  with  div(ep*u) = 0
//
// Inputs:
// 
//   u_mac,v_mac,w_mac = the MAC velocity field to be projected
//   density           = the cell-centered density
//
// Outputs:
//
//  u_mac,v_mac,w_mac = the PROJECTED MAC velocity field 
//
// Notes:
//
//  phi, the projection auxiliary function, is computed by solving
//
//       div(ep*grad(phi)/rho) = div(ep * u*)
// 
void 
incflo::apply_MAC_projection (AMREX_D_DECL(Vector<MultiFab*> const& u_mac,
                                           Vector<MultiFab*> const& v_mac,
                                           Vector<MultiFab*> const& w_mac),
                              Vector<Array<MultiFab const*,AMREX_SPACEDIM>> const& inv_rho,
                              Real time)
{
    BL_PROFILE("incflo::apply_MAC_projection()");

    if (m_verbose > 2) amrex::Print() << "MAC Projection:\n";

    Vector<Array<MultiFab*,AMREX_SPACEDIM> > mac_vec(finest_level+1);
    for (int lev=0; lev <= finest_level; ++lev)
    {
        AMREX_D_TERM(mac_vec[lev][0] = u_mac[lev];,
                     mac_vec[lev][1] = v_mac[lev];,
                     mac_vec[lev][2] = w_mac[lev];);
    }

    //
    // If we want to set max_coarsening_level we have to send it in to the constructor
    //
    LPInfo lp_info;
    lp_info.setMaxCoarseningLevel(m_mac_mg_max_coarsening_level);

    //
    // Perform MAC projection
    //
#if AMREX_USE_EB
    MacProjector macproj(mac_vec, MLMG::Location::FaceCentroid, // Location of mac_vec   
                         inv_rho, MLMG::Location::FaceCentroid, // Location of beta   
                                  MLMG::Location::CellCenter  , // Location of solution variable phi
                         Geom(0,finest_level), lp_info);
#else
    MacProjector macproj(mac_vec,inv_rho,Geom(0,finest_level), lp_info);
#endif

    macproj.setDomainBC(get_projection_bc(Orientation::low), get_projection_bc(Orientation::high));
   
    auto mac_phi = get_mac_phi();

    if (m_use_mac_phi_in_godunov)
    {
        for (int lev=0; lev <= finest_level; ++lev)
            mac_phi[lev]->mult(m_dt/2.,0,1,1);

        macproj.project(mac_phi,m_mac_mg_rtol,m_mac_mg_atol);

        for (int lev=0; lev <= finest_level; ++lev)
            mac_phi[lev]->mult(2./m_dt,0,1,1);
    } else {
        macproj.project(m_mac_mg_rtol,m_mac_mg_atol);
    }
}
