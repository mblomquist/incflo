#include <incflo.H>

using namespace amrex;

void incflo::prob_init_fluid (int lev)
{
    auto& ld = *m_leveldata[lev];
    Box const& domain = geom[lev].Domain();
    auto const& dx = geom[lev].CellSizeArray();
    auto const& problo = geom[lev].ProbLoArray();
    auto const& probhi = geom[lev].ProbHiArray();

    ld.p_nd.setVal(0.0);
    ld.gp.setVal(0.0);

    ld.density.setVal(m_ro_0);
    ld.density_o.setVal(m_ro_0);

    AMREX_D_TERM(ld.velocity.setVal(m_ic_u, 0, 1);,
                 ld.velocity.setVal(m_ic_v, 1, 1);,
                 ld.velocity.setVal(m_ic_w, 2, 1););

    if (m_ntrac > 0) ld.tracer.setVal(0.0);
 
    amrex::Print() <<" TYPE " << m_probtype << std::endl;

    for (MFIter mfi(ld.density); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        const Box& gbx = mfi.fabbox();
        if (0 == m_probtype || 114 == m_probtype )
        { }
        else if (1 == m_probtype)
        {
            init_taylor_green(vbx, gbx,
                              ld.velocity.array(mfi),
                              ld.density.array(mfi),
                              ld.tracer.array(mfi),
                              domain, dx, problo, probhi);
        }
        else if (2 == m_probtype)
        {
            init_taylor_vortex(vbx, gbx,
                               ld.velocity.array(mfi),
                               ld.density.array(mfi),
                               ld.tracer.array(mfi),
                               domain, dx, problo, probhi);
        }
        else if (3 == m_probtype)
        {
            init_taylor_green3d(vbx, gbx,
                                ld.velocity.array(mfi),
                                ld.density.array(mfi),
                                ld.tracer.array(mfi),
                                domain, dx, problo, probhi);
        }
        else if (4 == m_probtype)
        {
            init_couette(vbx, gbx,
                         ld.velocity.array(mfi),
                         ld.density.array(mfi),
                         ld.tracer.array(mfi),
                         domain, dx, problo, probhi);
        }
        else if (5 == m_probtype)
        {
            init_rayleigh_taylor(vbx, gbx,
                                 ld.velocity.array(mfi),
                                 ld.density.array(mfi),
                                 ld.tracer.array(mfi),
                                 domain, dx, problo, probhi);
        }
        else if (6 == m_probtype)
        {
            init_channel_slant(vbx, gbx,
                               ld.velocity.array(mfi),
                               ld.density.array(mfi),
                               ld.tracer.array(mfi),
                               domain, dx, problo, probhi);
        }
        else if (11 == m_probtype)
        {
            init_tuscan(vbx, gbx,
                        ld.velocity.array(mfi),
                        ld.density.array(mfi),
                        ld.tracer.array(mfi),
                        domain, dx, problo, probhi);
        }
        else if (111 == m_probtype || 112 == m_probtype || 113 == m_probtype)
        {
            init_boussinesq_bubble(vbx, gbx,
                                   ld.velocity.array(mfi),
                                   ld.density.array(mfi),
                                   ld.tracer.array(mfi),
                                   domain, dx, problo, probhi);
        }
        else if (12 == m_probtype)
        {
            init_periodic_tracer(vbx, gbx,
                                 ld.velocity.array(mfi),
                                 ld.density.array(mfi),
                                 ld.tracer.array(mfi),
                                 domain, dx, problo, probhi);
        }
        else if (13 == m_probtype)
        {
            init_flow_in_box(vbx, gbx,
                             ld.velocity.array(mfi),
                             ld.density.array(mfi),
                             ld.tracer.array(mfi),
                             domain, dx, problo, probhi);
        }
        else if (66 == m_probtype)
        {
            init_vortex_in_sphere(vbx, gbx,
                                  ld.velocity.array(mfi),
                                  ld.density.array(mfi),
                                  ld.tracer.array(mfi),
                                  domain, dx, problo, probhi);
        }
        else if (21 == m_probtype || 22 == m_probtype || 23 == m_probtype)
        {
            init_double_shear_layer(vbx, gbx,
                                    ld.velocity.array(mfi),
                                    ld.density.array(mfi),
                                    ld.tracer.array(mfi),
                                    domain, dx, problo, probhi);
        }
        else if (31  == m_probtype || 32  == m_probtype || 33  == m_probtype ||
                 311 == m_probtype || 322 == m_probtype || 333 == m_probtype ||
                 41  == m_probtype)
        {
            init_plane_poiseuille(vbx, gbx,
                                  ld.velocity.array(mfi),
                                  ld.density.array(mfi),
                                  ld.tracer.array(mfi),
                                  domain, dx, problo, probhi);
        }
#if 0
        else if (500 == m_probtype)
        {
            init_heated_ground(vbx, gbx,
                               ld.velocity.array(mfi),
                               ld.density.array(mfi),
                               ld.tracer.array(mfi),
                               domain, dx, problo, probhi);
        }
#endif
        else
        {
            amrex::Abort("prob_init_fluid: unknown m_probtype");
        };
    }
}

void incflo::init_taylor_green (Box const& vbx, Box const& /*gbx*/,
                                Array4<Real> const& vel,
                                Array4<Real> const& /*density*/,
                                Array4<Real> const& /*tracer*/,
                                Box const& /*domain*/,
                                GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = Real(i+0.5)*dx[0];
        Real y = Real(j+0.5)*dx[1];
        constexpr Real twopi = Real(2.0)*Real(3.1415926535897932);
        vel(i,j,k,0) =  std::sin(twopi*x) * std::cos(twopi*y);
        vel(i,j,k,1) = -std::cos(twopi*x) * std::sin(twopi*y);
#if (AMREX_SPACEDIM == 3)
        vel(i,j,k,2) = Real(0.0);
#endif
    });
}

void incflo::init_taylor_green3d (Box const& vbx, Box const& /*gbx*/,
                                  Array4<Real> const& vel,
                                  Array4<Real> const& /*density*/,
                                  Array4<Real> const& /*tracer*/,
                                  Box const& /*domain*/,
                                  GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                  GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                  GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = Real(i+0.5)*dx[0];
        Real y = Real(j+0.5)*dx[1];
        Real z = Real(k+0.5)*dx[2];
        constexpr Real twopi = Real(2.0)*Real(3.1415926535897932);
        AMREX_D_TERM(vel(i,j,k,0) =  std::sin(twopi*x) * std::cos(twopi*y) * std::cos(twopi*z);,
                     vel(i,j,k,1) = -std::cos(twopi*x) * std::sin(twopi*y) * std::cos(twopi*z);,
                     vel(i,j,k,2) = 0.0;);
    });
}

void incflo::init_taylor_vortex (Box const& vbx, Box const& /*gbx*/,
                                 Array4<Real> const& vel,
                                 Array4<Real> const& /*density*/,
                                 Array4<Real> const& /*tracer*/,
                                 Box const& /*domain*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                 GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = Real(i+0.5)*dx[0];
        Real y = Real(j+0.5)*dx[1];
        constexpr Real pi = Real(3.1415926535897932);
        constexpr Real u0 = Real(1.0);
        constexpr Real v0 = Real(1.0);
        vel(i,j,k,0) =  u0 - std::cos(pi*x) * std::sin(pi*y);
        vel(i,j,k,1) =  v0 + std::sin(pi*x) * std::cos(pi*y);
#if (AMREX_SPACEDIM == 3)
        vel(i,j,k,2) = Real(0.0);
#endif
    });
}


void incflo::init_vortex_in_sphere (Box const& vbx, Box const& /*gbx*/,
                               Array4<Real> const& vel,
                               Array4<Real> const& /*density*/,
                               Array4<Real> const& /*tracer*/,
                               Box const& /*domain*/,
                               GpuArray<Real, AMREX_SPACEDIM> const& dx,
                               GpuArray<Real, AMREX_SPACEDIM> const& problo,
                               GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = problo[0] + Real(i+0.5)*dx[0];
        Real y = problo[1] + Real(j+0.5)*dx[1];
        Real deltax = x;
        Real deltay = y;
        Real d_sq = deltax*deltax + deltay*deltay;
        Real r_sq = Real(0.003)*Real(0.003);
        Real u_vort = -Real(0.2)*deltay/r_sq * std::exp(-d_sq/r_sq/Real(2.0));
        Real v_vort =  Real(0.2)*deltax/r_sq * std::exp(-d_sq/r_sq/Real(2.0));
        vel(i,j,k,0) =  u_vort;
        vel(i,j,k,1) =  v_vort;
#if (AMREX_SPACEDIM == 3)
        vel(i,j,k,2) = Real(0.0);
#endif
    });
}

void incflo::init_flow_in_box (Box const& vbx, Box const& /*gbx*/,
                               Array4<Real> const& vel,
                               Array4<Real> const& /*density*/,
                               Array4<Real> const& tracer,
                               Box const& /*domain*/,
                               GpuArray<Real, AMREX_SPACEDIM> const& dx,
                               GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                               GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    ParmParse pp("box");

    Vector<Real> boxLo(AMREX_SPACEDIM), boxHi(AMREX_SPACEDIM);
    Real offset = Real(1.0e-15);

    for(int i = 0; i < AMREX_SPACEDIM; i++)
    {
        boxLo[i] = geom[0].ProbLo(i);
        boxHi[i] = geom[0].ProbHi(i);
    }

    pp.queryarr("Lo", boxLo, 0, AMREX_SPACEDIM);
    pp.queryarr("Hi", boxHi, 0, AMREX_SPACEDIM);

#if (AMREX_SPACEDIM == 3)
    int periodic_dir;
    pp.get("periodic_dir", periodic_dir);
#endif

    pp.query("offset", offset);

    Real xlo = boxLo[0] + offset;
    Real xhi = boxHi[0] - offset;

    Real ylo = boxLo[1] + offset;
    Real yhi = boxHi[1] - offset;

#if (AMREX_SPACEDIM == 3)
    Real zlo = boxLo[2] + offset;
    Real zhi = boxHi[2] - offset;
#endif

#if (AMREX_SPACEDIM == 3)
    if (periodic_dir == 0)
    {
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real y = Real(j+0.5)*dx[1]*(yhi-ylo) + ylo;
            Real z = Real(k+0.5)*dx[2]*(zhi-zlo) + zlo;
            constexpr Real pi = Real(3.1415926535897932);
            vel(i,j,k,1) =  std::sin(pi*y) * std::cos(pi*z);
            vel(i,j,k,2) = -std::cos(pi*y) * std::sin(pi*z);
            vel(i,j,k,0) = Real(1.0);
            if (y < 0.5)
                tracer(i,j,k) = Real(0.0);
            else
                tracer(i,j,k) = Real(1.0);
        });
    } else if (periodic_dir == 1)
    {
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = Real(i+0.5)*dx[0]*(xhi-xlo) + xlo;
            Real z = Real(k+0.5)*dx[2]*(zhi-zlo) + zlo;
            constexpr Real pi = Real(3.1415926535897932);
            vel(i,j,k,2) =  std::sin(pi*z) * std::cos(pi*x);
            vel(i,j,k,0) = -std::cos(pi*z) * std::sin(pi*x);
            vel(i,j,k,1) = Real(1.0);
            if (z < Real(0.5))
                tracer(i,j,k) = Real(0.0);
            else
                tracer(i,j,k) = Real(1.0);
        });
    } else if (periodic_dir == 2)
#endif
    {
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = Real(i+0.5)*dx[0]*(xhi-xlo) + xlo;
            Real y = Real(j+0.5)*dx[1]*(yhi-ylo) + ylo;
            constexpr Real pi = Real(3.1415926535897932);
            vel(i,j,k,0) =  std::sin(pi*x) * std::cos(pi*y);
            vel(i,j,k,1) = -std::cos(pi*x) * std::sin(pi*y);
#if (AMREX_SPACEDIM == 3)
            vel(i,j,k,2) = Real(1.0);
#endif
            if (x < Real(0.5))
                tracer(i,j,k) = Real(0.0);
            else
                tracer(i,j,k) = Real(1.0);
        });
#if (AMREX_SPACEDIM == 3)
    } else {
       amrex::Error("flow_in_box assumes a periodic direction");
#endif
    }
}

void incflo::init_couette (Box const& vbx, Box const& /*gbx*/,
                           Array4<Real> const& vel,
                           Array4<Real> const& /*density*/,
                           Array4<Real> const& /*tracer*/,
                           Box const& domain,
                           GpuArray<Real, AMREX_SPACEDIM> const& /*dx*/,
                           GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                           GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    Real num_cells_y = static_cast<Real>(domain.length(1));
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real y = Real(j+0.5) / num_cells_y;
        AMREX_D_TERM(vel(i,j,k,0) *= (y-Real(0.5));,
                     vel(i,j,k,1) = Real(0.0);,
                     vel(i,j,k,2) = Real(0.0););
    });
}

void incflo::init_channel_slant (Box const& vbx, Box const& /*gbx*/,
                                 Array4<Real> const& /*vel*/,
                                 Array4<Real> const& density,
                                 Array4<Real> const& tracer,
                                 Box const& domain,
                                 GpuArray<Real, AMREX_SPACEDIM> const& /*dx*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    const auto dhi = amrex::ubound(domain);
    int direction  = -1;

    // Get cylinder information from inputs file.                               *
    ParmParse pp("cylinder");
    pp.get("direction",  direction);

    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (density(i,j,k)>0) {

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n)
                tracer(i,j,k,n) = Real(0.0);

            if (direction == 0) {
                if (nt > 0 && i <= dhi.x/8)   tracer(i,j,k,0) = Real(1.0);
                if (nt > 1 && i <= dhi.x/2)   tracer(i,j,k,1) = Real(2.0);
                if (nt > 2 && i <= dhi.x*3/4) tracer(i,j,k,2) = Real(3.0);
            } else if (direction == 1) {
                if (nt > 0 && j <= dhi.y/8)   tracer(i,j,k,0) = Real(1.0);
                if (nt > 1 && j <= dhi.y/2)   tracer(i,j,k,1) = Real(2.0);
                if (nt > 2 && j <= dhi.y*3/4) tracer(i,j,k,2) = Real(3.0);
#if (AMREX_SPACEDIM == 3)
            } else {
                if (nt > 0 && k <= dhi.z/8)   tracer(i,j,k,0) = Real(1.0);
                if (nt > 1 && k <= dhi.z/2)   tracer(i,j,k,1) = Real(2.0);
                if (nt > 2 && k <= dhi.z*3/4) tracer(i,j,k,2) = Real(3.0);
#endif
            }
        }
    });
}

void incflo::init_rayleigh_taylor (Box const& vbx, Box const& /*gbx*/,
                                   Array4<Real> const& vel,
                                   Array4<Real> const& density,
                                   Array4<Real> const& tracer,
                                   Box const& /*domain*/,
                                   GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                   GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                   GpuArray<Real, AMREX_SPACEDIM> const& probhi)
{
    static constexpr Real pi    = Real(3.1415926535897932);
    static constexpr Real rho_1 = Real(0.5);
    static constexpr Real rho_2 = Real(2.0);
    static constexpr Real tra_1 = Real(0.0);
    static constexpr Real tra_2 = Real(1.0);

    static constexpr Real width = Real(0.005);

    const Real splitx = Real(0.5)*(problo[0] + probhi[0]);
    const Real L_x    = probhi[0] - problo[0];

#if (AMREX_SPACEDIM == 2)
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        vel(i,j,k,0) = Real(0.0);
        vel(i,j,k,1) = Real(0.0);

        Real x = problo[0] + Real(i+0.5)*dx[0];
        Real y = problo[1] + Real(j+0.5)*dx[1];

        const Real r2d = amrex::min(std::abs(x-splitx), Real(0.5)*L_x);
        const Real pertheight = Real(0.5) - Real(0.01)*std::cos(Real(2.0)*pi*r2d/L_x);

        density(i,j,k) = rho_1 + (Real(0.5)*(rho_2-rho_1))*(Real(1.0)+std::tanh((y-pertheight)/width));
        tracer(i,j,k)  = tra_1 + (Real(0.5)*(tra_2-tra_1))*(Real(1.0)+std::tanh((y-pertheight)/width));
    });

#elif (AMREX_SPACEDIM == 3)

    const Real splity = Real(0.5)*(problo[1] + probhi[1]);
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        vel(i,j,k,0) = Real(0.0);
        vel(i,j,k,1) = Real(0.0);
        vel(i,j,k,2) = Real(0.0);

        Real x = problo[0] + Real(i+0.5)*dx[0];
        Real y = problo[1] + Real(j+0.5)*dx[1];
        Real z = problo[2] + Real(k+0.5)*dx[2];

        const Real r2d = amrex::min(std::hypot((x-splitx),(y-splity)), Real(0.5)*L_x);
        const Real pertheight = Real(0.5) - Real(0.01)*std::cos(Real(2.0)*pi*r2d/L_x);

        density(i,j,k) = rho_1 + (Real(0.5)*(rho_2-rho_1))*(Real(1.0)+std::tanh((z-pertheight)/width));
        tracer(i,j,k)  = tra_1 + (Real(0.5)*(tra_2-tra_1))*(Real(1.0)+std::tanh((z-pertheight)/width));
    });
#endif
}

void incflo::init_tuscan (Box const& vbx, Box const& /*gbx*/,
                          Array4<Real> const& vel,
                          Array4<Real> const& density,
                          Array4<Real> const& tracer,
                          Box const& domain,
                          GpuArray<Real, AMREX_SPACEDIM> const& /*dx*/,
                          GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                          GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    int half_num_cells = domain.length(2) / 2;
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        AMREX_D_TERM(vel(i,j,k,0) = Real(0.0);,
                     vel(i,j,k,1) = Real(0.0);,
                     vel(i,j,k,2) = Real(0.0););
        density(i,j,k) = Real(1.0);
        if (k <= half_num_cells) {
            tracer(i,j,k) = Real(0.0);
        } else {
            tracer(i,j,k) = Real(0.01);
         }
    });
}

void incflo::init_boussinesq_bubble (Box const& vbx, Box const& /*gbx*/,
                                     Array4<Real> const& vel,
                                     Array4<Real> const& density,
                                     Array4<Real> const& tracer,
                                     Box const& /*domain*/,
                                     GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                     GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                     GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    if (111 == m_probtype)
    {
        constexpr Real m_fourth = Real(0.25);
        constexpr Real   m_half = Real(0.50);
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            vel(i,j,k,0) = 0.0;
            vel(i,j,k,1) = 0.0;
#if (AMREX_SPACEDIM == 3)
            vel(i,j,k,2) = 0.0;
#endif
            density(i,j,k) = 1.0;

            Real x = Real(i+0.5)*dx[0];
            Real y = Real(j+0.5)*dx[1];
#if (AMREX_SPACEDIM == 2)
            Real r = std::sqrt((x-0.25)*(x-0.25) + (y-0.5)*(y-0.5));
#elif  (AMREX_SPACEDIM == 3)
            Real z = Real(k+0.5)*dx[2];
            Real r = std::sqrt((x-m_half)*(x-m_half) + (y-m_fourth)*(y-m_fourth) + (z-m_fourth)*(z-m_fourth));
#endif
            if (r < Real(0.1))
                tracer(i,j,k,0) = Real(0.0);
            else
                tracer(i,j,k,0) = Real(0.01);
        });
    }
#if (AMREX_SPACEDIM == 3)
    else if (112 == m_probtype) {
        constexpr Real m_fourth = Real(0.25);
        constexpr Real   m_half = Real(0.50);
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            vel(i,j,k,0) = Real(0.0);
            vel(i,j,k,1) = Real(0.0);
            vel(i,j,k,2) = Real(0.0);
            density(i,j,k) = Real(1.0);

            Real x = Real(i+0.5)*dx[0];
            Real y = Real(j+0.5)*dx[1];
            Real z = Real(k+0.5)*dx[2];

            Real r = std::sqrt( (x-m_fourth)*(x-m_fourth) + (y-m_half )*(y-m_half)
                               +(z-m_fourth)*(z-m_fourth));

            if(r < Real(0.1))
                tracer(i,j,k,0) = Real(0.0);
            else
                tracer(i,j,k,0) = Real(0.01);
        });
    } else if (113 == m_probtype) {
        constexpr Real m_fourth = Real(0.25);
        constexpr Real   m_half = Real(0.50);
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            vel(i,j,k,0) = 0.0;
            vel(i,j,k,1) = 0.0;
            vel(i,j,k,2) = 0.0;
            density(i,j,k) = 1.0;

            Real x = Real(i+0.5)*dx[0];
            Real y = Real(j+0.5)*dx[1];
            Real z = Real(k+0.5)*dx[2];

            Real r = std::sqrt((x-m_fourth)*(x-m_fourth) + (y-m_fourth)*(y-m_fourth) + (z-m_half)*(z-m_half));

            if(r < Real(0.1))
                tracer(i,j,k,0) = Real(0.0);
            else
                tracer(i,j,k,0) = Real(0.01);
        });
    }
#endif
}

void incflo::init_periodic_tracer (Box const& vbx, Box const& /*gbx*/,
                                   Array4<Real> const& vel,
                                   Array4<Real> const& /*density*/,
                                   Array4<Real> const& tracer,
                                   Box const& /*domain*/,
                                   GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                   GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                   GpuArray<Real, AMREX_SPACEDIM> const& probhi)
{
    Real L = probhi[0]-problo[0];
    Real C = Real(2.0)*Real(3.1415926535897932) / L;
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        constexpr Real A = Real(1.0);
        Real x = Real(i+0.5)*dx[0];
        Real y = Real(j+0.5)*dx[1];
        Real z = Real(k+0.5)*dx[2];
        AMREX_D_TERM(vel(i,j,k,0) = Real(1.0);,
                     vel(i,j,k,1) = Real(0.1)*(std::sin(C*(x+z) - Real(0.00042)) + Real(1.0)) * std::exp(y);,
                     vel(i,j,k,2) = Real(0.1)*(std::sin(C*(x+y) - Real(0.00042)) + Real(1.0)) * std::exp(z););
        tracer(i,j,k) = A *(std::sin(C*(y+z) - Real(0.00042)) + Real(1.0)) * std::exp(x);
    });
}

void incflo::init_double_shear_layer (Box const& vbx, Box const& /*gbx*/,
                                      Array4<Real> const& vel,
                                      Array4<Real> const& /*density*/,
                                      Array4<Real> const& tracer,
                                      Box const& /*domain*/,
                                      GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                      GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                      GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    static constexpr Real twopi = Real(2.0) * Real(3.1415926535897932);
    if (21 == m_probtype)
    {
        constexpr Real m_fourth = Real(0.25);
        constexpr Real m_half   = Real(0.5);
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = Real(i+0.5) * dx[0];
            Real y = Real(j+0.5) * dx[1];
            vel(i,j,k,0) = std::tanh(Real(30.0)*(Real(0.25)-amrex::Math::abs(y-Real(0.5))));
            vel(i,j,k,1) = Real(0.05)*std::sin(twopi*x);
#if (AMREX_SPACEDIM == 3)
            vel(i,j,k,2) = Real(0.0);
#endif
            Real r = std::sqrt((x-m_half)*(x-m_half) + (y-m_fourth)*(y-m_fourth));
            if (r < Real(0.1))
                tracer(i,j,k,0) = Real(0.0);
            else
                tracer(i,j,k,0) = Real(0.01);
        });
    }
#if (AMREX_SPACEDIM == 3)
    else if (22 == m_probtype)
    {
        constexpr Real m_half = Real(0.5);
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real y = Real(j+0.5) * dx[1];
            Real z = Real(k+0.5) * dx[2];
            vel(i,j,k,1) = std::tanh(Real(30.0)*(Real(0.25)-amrex::Math::abs(z-m_half)));
            vel(i,j,k,2) = Real(0.05)*std::sin(twopi*y);
            vel(i,j,k,0) = Real(0.0);

            Real r = std::sqrt((y-m_half)*(y-m_half) + (z-m_half)*(z-m_half));
            if (r < Real(0.1))
                tracer(i,j,k,0) = Real(0.0);
            else
                tracer(i,j,k,0) = Real(0.01);
        });
    }
    else if (23 == m_probtype)
    {
        constexpr Real m_half = Real(0.5);
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = Real(i+0.5) * dx[0];
            Real z = Real(k+0.5) * dx[2];
            vel(i,j,k,2) = std::tanh(Real(30.0)*(Real(0.25)-amrex::Math::abs(x-m_half)));
            vel(i,j,k,0) = Real(0.05)*std::sin(twopi*z);
            vel(i,j,k,1) = Real(0.0);

            Real r = std::sqrt((x-m_half)*(x-m_half) + (z-m_half)*(z-m_half));
            if (r < .1)
                tracer(i,j,k,0) = Real(0.0);
            else
                tracer(i,j,k,0) = Real(0.01);
        });
    }
#endif
    else
    {
        amrex::Abort("Unknown double shear layer m_probtype");
    };
}

void incflo::init_plane_poiseuille (Box const& vbx, Box const& /*gbx*/,
                                    Array4<Real> const& vel,
                                    Array4<Real> const& /*density*/,
                                    Array4<Real> const& tracer,
                                    Box const& domain,
                                    GpuArray<Real, AMREX_SPACEDIM> const& /*dx*/,
                                    GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                    GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    Real dxinv = Real(1.0) / domain.length(0);
    Real dyinv = Real(1.0) / domain.length(1);
#if (AMREX_SPACEDIM == 3)
    Real dzinv = Real(1.0) / domain.length(2);
#else
    Real dzinv = 0.0;
#endif
    const auto dhi = amrex::ubound(domain);

    if (31 == m_probtype)
    {
        Real u = m_ic_u;
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real y = Real(j+0.5)*dyinv;
            AMREX_D_TERM(vel(i,j,k,0) = Real(6.0) * u * y * (Real(1.0)-y);,
                         vel(i,j,k,1) = Real(0.0);,
                         vel(i,j,k,2) = Real(0.0););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && i <= dhi.x/8)   tracer(i,j,k,0) = 1.0;
            if (nt > 1 && i <= dhi.x/2)   tracer(i,j,k,1) = 2.0;
            if (nt > 2 && i <= dhi.x*3/4) tracer(i,j,k,2) = 3.0;
        });
    }
    else if (311 == m_probtype)
    {
        Real u = m_ic_u;
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real z = Real(k+0.5)*dzinv;
            AMREX_D_TERM(vel(i,j,k,0) = Real(6.0) * u * z * (Real(1.0)-z);,
                         vel(i,j,k,1) = Real(0.0);,
                         vel(i,j,k,2) = Real(0.0););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && i <= dhi.x/8)   tracer(i,j,k,0) = 1.0;
            if (nt > 1 && i <= dhi.x/2)   tracer(i,j,k,1) = 2.0;
            if (nt > 2 && i <= dhi.x*3/4) tracer(i,j,k,2) = 3.0;
        });
    }
    else if (41 == m_probtype)
    {
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real z = Real(k+0.5)*dzinv;
            AMREX_D_TERM(vel(i,j,k,0) = Real(0.5)*z;,
                         vel(i,j,k,1) = Real(0.0);,
                         vel(i,j,k,2) = Real(0.0););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && i <= dhi.x/8)   tracer(i,j,k,0) = 1.0;
            if (nt > 1 && i <= dhi.x/2)   tracer(i,j,k,1) = 2.0;
            if (nt > 2 && i <= dhi.x*3/4) tracer(i,j,k,2) = 3.0;
        });
    }
    else if (32 == m_probtype)
    {
        Real v = m_ic_v;
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real z = Real(k+0.5)*dzinv;
            AMREX_D_TERM(vel(i,j,k,0) = Real(0.0);,
                         vel(i,j,k,1) = Real(6.0) * v * z * (Real(1.0)-z);,
                         vel(i,j,k,2) = Real(0.0););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && j <= dhi.y/8)   tracer(i,j,k,0) = 1.0;
            if (nt > 1 && j <= dhi.y/2)   tracer(i,j,k,1) = 2.0;
            if (nt > 2 && j <= dhi.y*3/4) tracer(i,j,k,2) = 3.0;
        });
    }
    else if (322 == m_probtype)
    {
        Real v = m_ic_v;
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = Real(i+0.5)*dxinv;
            AMREX_D_TERM(vel(i,j,k,0) = Real(0.0);,
                         vel(i,j,k,1) = Real(6.0) * v * x * (Real(1.0)-x);,
                         vel(i,j,k,2) = Real(0.0););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && j <= dhi.y/8)   tracer(i,j,k,0) = Real(1.0);
            if (nt > 1 && j <= dhi.y/2)   tracer(i,j,k,1) = Real(2.0);
            if (nt > 2 && j <= dhi.y*3/4) tracer(i,j,k,2) = Real(3.0);
        });
    }
    else if (33 == m_probtype)
    {
#if (AMREX_SPACEDIM == 3)
        Real w = m_ic_w;
#endif
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
#if (AMREX_SPACEDIM == 3)
            Real x = Real(i+0.5)*dxinv;
#endif
            AMREX_D_TERM(vel(i,j,k,0) = 0.0;,
                         vel(i,j,k,1) = 0.0;,
                         vel(i,j,k,2) = Real(6.0) * w * x * (Real(1.0)-x););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && k <= dhi.z/8)   tracer(i,j,k,0) = Real(1.0);
            if (nt > 1 && k <= dhi.z/2)   tracer(i,j,k,1) = Real(2.0);
            if (nt > 2 && k <= dhi.z*3/4) tracer(i,j,k,2) = Real(3.0);
        });
    }
    else if (333 == m_probtype)
    {
#if (AMREX_SPACEDIM == 3)
        Real w = m_ic_w;
#endif
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
#if (AMREX_SPACEDIM == 3)
            Real y = Real(j+0.5)*dyinv;
#endif
            AMREX_D_TERM(vel(i,j,k,0) = 0.0;,
                         vel(i,j,k,1) = 0.0;,
                         vel(i,j,k,2) = Real(6.0) * w * y * (Real(1.0)-y););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && k <= dhi.z/8)   tracer(i,j,k,0) = Real(1.0);
            if (nt > 1 && k <= dhi.z/2)   tracer(i,j,k,1) = Real(2.0);
            if (nt > 2 && k <= dhi.z*3/4) tracer(i,j,k,2) = Real(3.0);
        });
    }
    else
    {
        amrex::Abort("Unknown plane poiseuille m_probtype");
    };
}
