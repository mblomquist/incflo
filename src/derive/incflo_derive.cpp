#include <AMReX_Box.H>

#include <incflo.H>
#include <incflo_derive_K.H>

using namespace amrex;

void incflo::ComputeDivU(Real /*time_in*/)
{
#if 0

    incflo_set_velocity_bcs(time_in, vel);

    int bc_lo[3], bc_hi[3];
    Box domain(geom[0].Domain());

    set_ppe_bcs(bc_lo, bc_hi,
                domain.loVect(), domain.hiVect(),
                &nghost,
                bc_ilo[0]->dataPtr(), bc_ihi[0]->dataPtr(),
                bc_jlo[0]->dataPtr(), bc_jhi[0]->dataPtr(),
                bc_klo[0]->dataPtr(), bc_khi[0]->dataPtr());

    ppe_lobc = {(LinOpBCType)bc_lo[0], (LinOpBCType)bc_lo[1], (LinOpBCType)bc_lo[2]};
    ppe_hibc = {(LinOpBCType)bc_hi[0], (LinOpBCType)bc_hi[1], (LinOpBCType)bc_hi[2]};

    LPInfo lpinfo;

    //
    // This rebuilds integrals each time linop is created -- must find a better way
    //

#ifdef AMREX_USE_EB
    MLNodeLaplacian linop(geom, grids, dmap, lpinfo, GetVecOfConstPtrs(ebfactory));
#else
    MLNodeLaplacian linop(geom, grids, dmap, lpinfo);
#endif
    linop.setDomainBC(ppe_lobc,ppe_hibc);
    linop.compDivergence(GetVecOfPtrs(divu),GetVecOfPtrs(vel));
#endif
}

#ifdef AMREX_USE_EB
void incflo::compute_strainrate_at_level (int lev,
#else
void incflo::compute_strainrate_at_level (int /*lev*/,
#endif
                                          MultiFab* strainrate,
                                          MultiFab* vel,
                                          Geometry& lev_geom,
                                          Real /*time*/, int nghost)
{

#ifdef AMREX_USE_EB
        auto const& fact = EBFactory(lev);
        auto const& flags = fact.getMultiEBCellFlagFab();
#endif

        AMREX_D_TERM(Real idx = Real(1.0) / lev_geom.CellSize(0);,
                     Real idy = Real(1.0) / lev_geom.CellSize(1);,
                     Real idz = Real(1.0) / lev_geom.CellSize(2););

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*strainrate,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
                Box const& bx = mfi.growntilebox(nghost);
                Array4<Real> const& sr_arr = strainrate->array(mfi);
                Array4<Real const> const& vel_arr = vel->const_array(mfi);
#ifdef AMREX_USE_EB
                auto const& flag_fab = flags[mfi];
                auto typ = flag_fab.getType(bx);
                if (typ == FabType::covered)
                {
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        sr_arr(i,j,k) = Real(0.0);
                    });
                }
                else if (typ == FabType::singlevalued)
                {
                    auto const& flag_arr = flag_fab.const_array();
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        sr_arr(i,j,k) = incflo_strainrate_eb(i,j,k,AMREX_D_DECL(idx,idy,idz),vel_arr,flag_arr(i,j,k));
                    });
                }
                else
#endif
                {
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        sr_arr(i,j,k) = incflo_strainrate(i,j,k,AMREX_D_DECL(idx,idy,idz),vel_arr);
                    });
                }
        }
}

Real incflo::ComputeKineticEnergy () const
{
#if 0
    BL_PROFILE("incflo::ComputeKineticEnergy");

    // integrated total Kinetic energy
    Real KE = Real(0.0);

    for(int lev = 0; lev <= finest_level; lev++)
    {
        Real cell_vol = geom[lev].CellSize()[0]*geom[lev].CellSize()[1]*geom[lev].CellSize()[2];

        KE += amrex::ReduceSum(*density[lev],*vel[lev],*level_mask[lev],0,
        [=] AMREX_GPU_HOST_DEVICE (Box const& bx,
                                   Array4<Real const> const& den_arr,
                                   Array4<Real const> const& vel_arr,
                                   Array4<int const>  const& mask_arr) -> Real
        {
            Real KE_Fab = Real(0.0);

            amrex::Loop(bx, [=,&KE_Fab] (int i, int j, int k) noexcept
            {
                KE_Fab += cell_vol*mask_arr(i,j,k)*den_arr(i,j,k)*( vel_arr(i,j,k,0)*vel_arr(i,j,k,0)
                                                                   +vel_arr(i,j,k,1)*vel_arr(i,j,k,1)
                                                                   +vel_arr(i,j,k,2)*vel_arr(i,j,k,2));

            });
            return KE_Fab;

        });
    }

    // total volume of grid on level 0
    Real total_vol = geom[0].ProbDomain().volume();

    KE *= Real(0.5)/total_vol/ro_0;

    ParallelDescriptor::ReduceRealSum(KE);

    return KE;

#endif
    return 0;
}

#if (AMREX_SPACEDIM == 2)
void incflo::ComputeVorticity (int lev, Real /*time*/, MultiFab& vort, MultiFab const& vel)
{
    BL_PROFILE("incflo::ComputeVorticity");
    const Real idx = Geom(lev).InvCellSize(0);
    const Real idy = Geom(lev).InvCellSize(1);

#ifdef AMREX_USE_EB
    const auto& fact = EBFactory(lev);
    const auto& flags_mf = fact.getMultiEBCellFlagFab();
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for(MFIter mfi(vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box bx = mfi.tilebox();
        Array4<Real const> const& ccvel_fab = vel.const_array(mfi);
        Array4<Real> const& vort_fab = vort.array(mfi);

#ifdef AMREX_USE_EB
        const EBCellFlagFab& flags = flags_mf[mfi];
        auto typ = flags.getType(bx);
        if (typ == FabType::covered)
        {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                vort_fab(i,j,k) = Real(0.0);
            });
        }
        else if (typ == FabType::singlevalued)
        {
            const auto& flag_fab = flags.const_array();
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                constexpr Real c0 = Real(-1.5);
                constexpr Real c1 = Real( 2.0);
                constexpr Real c2 = Real(-0.5);

                if (flag_fab(i,j,k).isCovered())
                {
                    vort_fab(i,j,k) = Real(0.0);
                }
                else
                {
                    Real vx, uy;
                    // Need to check if there are covered cells in neighbours --
                    // -- if so, use one-sided difference computation (but still quadratic)
                    if (!flag_fab(i,j,k).isConnected( 1,0,0))
                    {
                        // Covered cell to the right, go fish left
                        vx = - (c0 * ccvel_fab(i  ,j,k,1)
                              + c1 * ccvel_fab(i-1,j,k,1)
                              + c2 * ccvel_fab(i-2,j,k,1)) * idx;
                    }
                    else if (!flag_fab(i,j,k).isConnected(-1,0,0))
                    {
                        // Covered cell to the left, go fish right
                        vx = (c0 * ccvel_fab(i  ,j,k,1)
                            + c1 * ccvel_fab(i+1,j,k,1)
                            + c2 * ccvel_fab(i+2,j,k,1)) * idx;
                    }
                    else
                    {
                        // No covered cells right or left, use standard stencil
                        vx = Real(0.5) * (ccvel_fab(i+1,j,k,1) - ccvel_fab(i-1,j,k,1)) * idx;
                    }
                    // Do the same in y-direction
                    if (!flag_fab(i,j,k).isConnected(0, 1,0))
                    {
                        uy = - (c0 * ccvel_fab(i,j  ,k,0)
                              + c1 * ccvel_fab(i,j-1,k,0)
                              + c2 * ccvel_fab(i,j-2,k,0)) * idy;
                    }
                    else if (!flag_fab(i,j,k).isConnected(0,-1,0))
                    {
                        uy = (c0 * ccvel_fab(i,j  ,k,0)
                            + c1 * ccvel_fab(i,j+1,k,0)
                            + c2 * ccvel_fab(i,j+2,k,0)) * idy;
                    }
                    else
                    {
                        uy = Real(0.5) * (ccvel_fab(i,j+1,k,0) - ccvel_fab(i,j-1,k,0)) * idy;
                    }
                    vort_fab(i,j,k) = vx-uy;
                }
            });
        }
        else
#endif
        {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real vx = Real(0.5) * (ccvel_fab(i+1,j,k,1) - ccvel_fab(i-1,j,k,1)) * idx;
                Real uy = Real(0.5) * (ccvel_fab(i,j+1,k,0) - ccvel_fab(i,j-1,k,0)) * idy;
                vort_fab(i,j,k) = vx-uy;
            });
        }
    }
}

#elif (AMREX_SPACEDIM == 3)
void incflo::ComputeVorticity (int lev, Real /*time*/, MultiFab& vort, MultiFab const& vel)
{
    BL_PROFILE("incflo::ComputeVorticity");
    const Real idx = Geom(lev).InvCellSize(0);
    const Real idy = Geom(lev).InvCellSize(1);
    const Real idz = Geom(lev).InvCellSize(2);

#ifdef AMREX_USE_EB
    const auto& fact = EBFactory(lev);
    const auto& flags_mf = fact.getMultiEBCellFlagFab();
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for(MFIter mfi(vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box bx = mfi.tilebox();
        Array4<Real const> const& ccvel_fab = vel.const_array(mfi);
        Array4<Real> const& vort_fab = vort.array(mfi);

#ifdef AMREX_USE_EB
        const EBCellFlagFab& flags = flags_mf[mfi];
        auto typ = flags.getType(bx);
        if (typ == FabType::covered)
        {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                vort_fab(i,j,k) = Real(0.0);
            });
        }
        else if (typ == FabType::singlevalued)
        {
            const auto& flag_fab = flags.const_array();
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                constexpr Real c0 = Real(-1.5);
                constexpr Real c1 = Real( 2.0);
                constexpr Real c2 = Real(-0.5);

                if (flag_fab(i,j,k).isCovered())
                {
                    vort_fab(i,j,k) = Real(0.0);
                }
                else
                {
                    Real vx, wx, uy, wy, uz, vz;
                    // Need to check if there are covered cells in neighbours --
                    // -- if so, use one-sided difference computation (but still quadratic)
                    if (!flag_fab(i,j,k).isConnected( 1,0,0))
                    {
                        // Covered cell to the right, go fish left
                        vx = - (c0 * ccvel_fab(i  ,j,k,1)
                              + c1 * ccvel_fab(i-1,j,k,1)
                              + c2 * ccvel_fab(i-2,j,k,1)) * idx;
                        wx = - (c0 * ccvel_fab(i  ,j,k,2)
                              + c1 * ccvel_fab(i-1,j,k,2)
                              + c2 * ccvel_fab(i-2,j,k,2)) * idx;
                    }
                    else if (!flag_fab(i,j,k).isConnected(-1,0,0))
                    {
                        // Covered cell to the left, go fish right
                        vx = (c0 * ccvel_fab(i  ,j,k,1)
                            + c1 * ccvel_fab(i+1,j,k,1)
                            + c2 * ccvel_fab(i+2,j,k,1)) * idx;
                        wx = (c0 * ccvel_fab(i  ,j,k,2)
                            + c1 * ccvel_fab(i+1,j,k,2)
                            + c2 * ccvel_fab(i+2,j,k,2)) * idx;
                    }
                    else
                    {
                        // No covered cells right or left, use standard stencil
                        vx = Real(0.5) * (ccvel_fab(i+1,j,k,1) - ccvel_fab(i-1,j,k,1)) * idx;
                        wx = Real(0.5) * (ccvel_fab(i+1,j,k,2) - ccvel_fab(i-1,j,k,2)) * idx;
                    }
                    // Do the same in y-direction
                    if (!flag_fab(i,j,k).isConnected(0, 1,0))
                    {
                        uy = - (c0 * ccvel_fab(i,j  ,k,0)
                              + c1 * ccvel_fab(i,j-1,k,0)
                              + c2 * ccvel_fab(i,j-2,k,0)) * idy;
                        wy = - (c0 * ccvel_fab(i,j  ,k,2)
                              + c1 * ccvel_fab(i,j-1,k,2)
                              + c2 * ccvel_fab(i,j-2,k,2)) * idy;
                    }
                    else if (!flag_fab(i,j,k).isConnected(0,-1,0))
                    {
                        uy = (c0 * ccvel_fab(i,j  ,k,0)
                            + c1 * ccvel_fab(i,j+1,k,0)
                            + c2 * ccvel_fab(i,j+2,k,0)) * idy;
                        wy = (c0 * ccvel_fab(i,j  ,k,2)
                            + c1 * ccvel_fab(i,j+1,k,2)
                            + c2 * ccvel_fab(i,j+2,k,2)) * idy;
                    }
                    else
                    {
                        uy = Real(0.5) * (ccvel_fab(i,j+1,k,0) - ccvel_fab(i,j-1,k,0)) * idy;
                        wy = Real(0.5) * (ccvel_fab(i,j+1,k,2) - ccvel_fab(i,j-1,k,2)) * idy;
                    }
                    // Do the same in z-direction
                    if (!flag_fab(i,j,k).isConnected(0,0, 1))
                    {
                        uz = - (c0 * ccvel_fab(i,j,k  ,0)
                              + c1 * ccvel_fab(i,j,k-1,0)
                              + c2 * ccvel_fab(i,j,k-2,0)) * idz;
                        vz = - (c0 * ccvel_fab(i,j,k  ,1)
                              + c1 * ccvel_fab(i,j,k-1,1)
                              + c2 * ccvel_fab(i,j,k-2,1)) * idz;
                    }
                    else if (!flag_fab(i,j,k).isConnected(0,0,-1))
                    {
                        uz = (c0 * ccvel_fab(i,j,k  ,0)
                            + c1 * ccvel_fab(i,j,k+1,0)
                            + c2 * ccvel_fab(i,j,k+2,0)) * idz;
                        vz = (c0 * ccvel_fab(i,j,k  ,1)
                            + c1 * ccvel_fab(i,j,k+1,1)
                            + c2 * ccvel_fab(i,j,k+2,1)) * idz;
                    }
                    else
                    {
                        uz = Real(0.5) * (ccvel_fab(i,j,k+1,0) - ccvel_fab(i,j,k-1,0)) * idz;
                        vz = Real(0.5) * (ccvel_fab(i,j,k+1,1) - ccvel_fab(i,j,k-1,1)) * idz;
                    }
                    vort_fab(i,j,k) = std::sqrt((wy-vz)*(wy-vz) + (uz-wx)*(uz-wx) + (vx-uy)*(vx-uy));
                }
            });
        }
        else
#endif
        {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real vx = Real(0.5) * (ccvel_fab(i+1,j,k,1) - ccvel_fab(i-1,j,k,1)) * idx;
                Real wx = Real(0.5) * (ccvel_fab(i+1,j,k,2) - ccvel_fab(i-1,j,k,2)) * idx;

                Real uy = Real(0.5) * (ccvel_fab(i,j+1,k,0) - ccvel_fab(i,j-1,k,0)) * idy;
                Real wy = Real(0.5) * (ccvel_fab(i,j+1,k,2) - ccvel_fab(i,j-1,k,2)) * idy;

                Real uz = Real(0.5) * (ccvel_fab(i,j,k+1,0) - ccvel_fab(i,j,k-1,0)) * idz;
                Real vz = Real(0.5) * (ccvel_fab(i,j,k+1,1) - ccvel_fab(i,j,k-1,1)) * idz;

                vort_fab(i,j,k) = std::sqrt((wy-vz)*(wy-vz) + (uz-wx)*(uz-wx) + (vx-uy)*(vx-uy));
            });
        }
    }
}
#endif

void incflo::ComputeDrag()
{
#if 0
    BL_PROFILE("incflo::ComputeDrag");

    // Coefficients for one-sided difference estimation
    Real c0 = -1.5;
    Real c1 = 2.0;
    Real c2 = -0.5;

    for(int lev = 0; lev <= finest_level; lev++)
    {
        Box domain(geom[lev].Domain());
        Real dx = geom[lev].CellSize()[0];

        drag[lev]->setVal(0.0);

#ifdef AMREX_USE_EB
        // Get EB geometric info
        const amrex::MultiCutFab* bndryarea;
        const amrex::MultiCutFab* bndrynorm;
        bndryarea = &(ebfactory[lev]->getBndryArea());
        bndrynorm = &(ebfactory[lev]->getBndryNormal());

#ifdef _OPENMP
#pragma omp parallel for reduction(+:drag) if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi(*vel[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // Tilebox
            Box bx = mfi.tilebox();

            // This is to check efficiently if this tile contains any eb stuff
            const EBFArrayBox& vel_fab = static_cast<EBFArrayBox const&>((*vel[lev])[mfi]);
            const EBCellFlagFab& flags = vel_fab.getEBCellFlagFab();

            if (flags.getType(bx) == FabType::singlevalued)
            {
                const auto& drag_arr = drag[lev]->array(mfi);
                const auto& vel_arr = vel[lev]->array(mfi);
                const auto& eta_arr = eta[lev]->array(mfi);
                const auto& p_arr = p[lev]->array(mfi);
                const auto& bndryarea_arr = bndryarea->array(mfi);
                const auto& bndrynorm_arr = bndrynorm->array(mfi);
                const auto& flag_fab = flags.array();

                for(int i = bx.smallEnd(0); i <= bx.bigEnd(0); i++)
                for(int j = bx.smallEnd(1); j <= bx.bigEnd(1); j++)
                for(int k = bx.smallEnd(2); k <= bx.bigEnd(2); k++)
                {
                    if(flag_fab(i,j,k).isSingleValued())
                    {
                        Real area = bndryarea_arr(i,j,k);
                        Real nx = bndrynorm_arr(i,j,k,0);
                        Real ny = bndrynorm_arr(i,j,k,1);
                        Real nz = bndrynorm_arr(i,j,k,2);

                        Real uz, vz, wx, wy, wz;

                        if (!flag_fab(i,j,k).isConnected(0,0, 1))
                        {
                            uz = - (c0 * vel_arr(i,j,k,0) + c1 * vel_arr(i,j,k-1,0) + c2 * vel_arr(i,j,k-2,0)) / dx;
                            vz = - (c0 * vel_arr(i,j,k,1) + c1 * vel_arr(i,j,k-1,1) + c2 * vel_arr(i,j,k-2,1)) / dx;
                            wz = - (c0 * vel_arr(i,j,k,2) + c1 * vel_arr(i,j,k-1,2) + c2 * vel_arr(i,j,k-2,2)) / dx;
                        }
                        else if (!flag_fab(i,j,k).isConnected(0,0,-1))
                        {
                            uz = (c0 * vel_arr(i,j,k,0) + c1 * vel_arr(i,j,k+1,0) + c2 * vel_arr(i,j,k+2,0)) / dx;
                            vz = (c0 * vel_arr(i,j,k,1) + c1 * vel_arr(i,j,k+1,1) + c2 * vel_arr(i,j,k+2,1)) / dx;
                            wz = (c0 * vel_arr(i,j,k,2) + c1 * vel_arr(i,j,k+1,2) + c2 * vel_arr(i,j,k+2,2)) / dx;
                        }
                        else
                        {
                            uz = 0.5 * (vel_arr(i,j,k+1,0) - vel_arr(i,j,k-1,0)) / dx;
                            vz = 0.5 * (vel_arr(i,j,k+1,1) - vel_arr(i,j,k-1,1)) / dx;
                            wz = 0.5 * (vel_arr(i,j,k+1,2) - vel_arr(i,j,k-1,2)) / dx;
                        }

                        if (!flag_fab(i,j,k).isConnected(0, 1,0))
                        {
                            wy = - (c0 * vel_arr(i,j,k,2) + c1 * vel_arr(i,j-1,k,2) + c2 * vel_arr(i,j-2,k,2)) / dx;
                        }
                        else if (!flag_fab(i,j,k).isConnected(0,-1,0))
                        {
                            wy = (c0 * vel_arr(i,j,k,2) + c1 * vel_arr(i,j+1,k,2) + c2 * vel_arr(i,j+2,k,2)) / dx;
                        }
                        else
                        {
                            wy = 0.5 * (vel_arr(i,j+1,k,2) - vel_arr(i,j-1,k,2)) / dx;
                        }

                        if (!flag_fab(i,j,k).isConnected( 1,0,0))
                        {
                            wx = - (c0 * vel_arr(i,j,k,2) + c1 * vel_arr(i-1,j,k,2) + c2 * vel_arr(i-2,j,k,2)) / dx;
                        }
                        else if (!flag_fab(i,j,k).isConnected(-1,0,0))
                        {
                            wx = (c0 * vel_arr(i,j,k,2) + c1 * vel_arr(i+1,j,k,2) + c2 * vel_arr(i+2,j,k,2)) / dx;
                        }
                        else
                        {
                            wx = 0.5 * (vel_arr(i+1,j,k,2) - vel_arr(i-1,j,k,2)) / dx;
                        }

                        Real p_contrib = p_arr(i,j,k) * nz;
                        Real tau_contrib = - eta_arr(i,j,k) * ( (uz + wx) * nx + (vz + wy) * ny + (wz + wz) * nz );

                        // TODO: Get values on EB centroid,
                        //       not the default CC and nodal values
                        drag_arr(i,j,k) = (p_contrib + tau_contrib) * area * dx * dx;
                    }
                    else
                    {
                        drag_arr(i,j,k) = 0.0;
                    }
                }
            }
        } // MFIter
#endif
    }
#endif
}
