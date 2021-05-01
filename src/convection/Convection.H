//
//  Convection.H
//

#ifndef Convection_H
#define Convection_H

#include <AMReX_BCRec.H>
#include <AMReX_MultiFab.H>

#ifdef AMREX_USE_EB
#include <AMReX_MultiCutFab.H>
#endif

namespace convection {

    void compute_convective_term (amrex::Box const& bx,
                                  amrex::Array4<amrex::Real> const& dbydt,
                                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& u_mac,
                                               amrex::Array4<amrex::Real const> const& v_mac,
                                               amrex::Array4<amrex::Real const> const& w_mac),
                                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& flux_x,
                                               amrex::Array4<amrex::Real const> const& flux_y,
                                               amrex::Array4<amrex::Real const> const& flux_z),
                                  int  const* d_iconserv,
                                  amrex::Geometry& geom);

    void compute_fluxes (amrex::Box const& bx, amrex::MFIter const& mfi,
                         amrex::Array4<amrex::Real const> const& vel,
                         amrex::Array4<amrex::Real const> const& rho,
                         amrex::Array4<amrex::Real const> const& tra,
                         amrex::Array4<amrex::Real const> const& divu,
                         AMREX_D_DECL(amrex::Array4<amrex::Real const> const& u_mac,
                                      amrex::Array4<amrex::Real const> const& v_mac,
                                      amrex::Array4<amrex::Real const> const& w_mac),
                         AMREX_D_DECL(amrex::Array4<amrex::Real> const& flux_x,
                                      amrex::Array4<amrex::Real> const& flux_y,
                                      amrex::Array4<amrex::Real> const& flux_z),
                         amrex::Array4<amrex::Real const> const& fv,
                         amrex::Array4<amrex::Real const> const& ft,
                         amrex::Vector<amrex::BCRec> const& h_vel_bcrec,
                                       amrex::BCRec  const* d_vel_bcrec,
                                                int  const* d_vel_iconserv,
                         amrex::Vector<amrex::BCRec> const& h_den_bcrec,
                                       amrex::BCRec  const* d_den_bcrec,
                                                int  const* d_den_iconserv,
                         amrex::Vector<amrex::BCRec> const& h_tra_bcrec,
                                       amrex::BCRec  const* d_tra_bcrec,
                                                int  const* d_tra_iconserv,
                         std::string l_advection_type, bool l_constant_density, 
                         bool l_advect_tracer, int l_ntrac,
                         bool l_godunov_ppm, bool l_godunov_use_forces_in_trans,
#ifdef AMREX_USE_EB
                         amrex::EBFArrayBoxFactory const* ebfact,
#endif
                         amrex::Geometry& geom,
                         amrex::Real l_dt);

#ifdef AMREX_USE_EB

    void redistribute_convective_term (amrex::Box const& bx, amrex::MFIter const& mfi,
                                       amrex::Array4<amrex::Real const> const& vel,       // velocity
                                       amrex::Array4<amrex::Real const> const& rho,       // density
                                       amrex::Array4<amrex::Real const> const& rhotrac,   // tracer
                                       amrex::Array4<amrex::Real> const& dvdt_tmp,  // initial velocity update
                                       amrex::Array4<amrex::Real> const& drdt_tmp,  // initial density update
                                       amrex::Array4<amrex::Real> const& dtdt_tmp,  // initial tracer update
                                       amrex::Array4<amrex::Real> const& dvdt,      // final velocity update
                                       amrex::Array4<amrex::Real> const& drdt,      // final density update
                                       amrex::Array4<amrex::Real> const& dtdt,      // final tracer update
                                       std::string l_redistribution_type, 
                                       bool l_constant_density, bool l_advect_tracer, int l_ntrac,
                                       amrex::EBFArrayBoxFactory const* ebfact,
                                       amrex::Geometry& geom, amrex::Real l_dt);
#endif
} // namespace convection

#endif /* Convection_H */
