#ifdef AMREX_USE_EB
#include <AMReX_EBMultiFabUtil.H>
#endif

#include <incflo.H>
#include <bc_mod_F.H>

using namespace amrex;

void
incflo::set_inflow_velocity (int lev, amrex::Real time, MultiFab& vel, int nghost)
{
    Geometry const& gm = Geom(lev);
    Box const& domain = gm.growPeriodicDomain(nghost);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        Orientation olo(dir,Orientation::low);
        Orientation ohi(dir,Orientation::high);
        if (m_bc_type[olo] == BC::mass_inflow or m_bc_type[ohi] == BC::mass_inflow) {
            Box dlo = (m_bc_type[olo] == BC::mass_inflow) ? amrex::adjCellLo(domain,dir,nghost) : Box();
            Box dhi = (m_bc_type[ohi] == BC::mass_inflow) ? amrex::adjCellHi(domain,dir,nghost) : Box();
#ifdef _OPENMP
#pragma omp parallel
#endif
            for (MFIter mfi(vel); mfi.isValid(); ++mfi) {
                Box const& gbx = amrex::grow(mfi.validbox(),nghost);
                Box blo = gbx & dlo;
                Box bhi = gbx & dhi;
                Array4<Real> const& v = vel[mfi].array();
                int gid = mfi.index();
                if (blo.ok()) {
                    prob_set_inflow_velocity(gid, olo, blo, v, lev, time);
                }
                if (bhi.ok()) {
                    prob_set_inflow_velocity(gid, ohi, bhi, v, lev, time);
                }
            }
        }
    }
}

//
//  These subroutines set the BCs for the vel_arr components only.
//

void
incflo::incflo_set_velocity_bcs (Real time,
                                 Vector< std::unique_ptr<MultiFab> > & vel_in) const
{
  BL_PROFILE("incflo::incflo_set_velocity_bcs()");

  for (int lev = 0; lev <= finest_level; lev++)
  {
     // Set all values outside the domain to covered_val just to avoid use of undefined
     vel_in[lev]->setDomainBndry(covered_val,geom[lev]);

     vel_in[lev] -> FillBoundary (geom[lev].periodicity());
     Box domain(geom[lev].Domain());

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
     for (MFIter mfi(*vel_in[lev]); mfi.isValid(); ++mfi) {
         set_velocity_bcs(time, lev, (*vel_in[lev])[mfi], domain);
     }

#ifdef AMREX_USE_EB
     EB_set_covered(*vel_in[lev], 0, vel_in[lev]->nComp(), vel_in[lev]->nGrow(), covered_val);
#endif

     // Do this after as well as before to pick up terms that got updated in the call above
     vel_in[lev] -> FillBoundary (geom[lev].periodicity());
  }
}

void
incflo::set_velocity_bcs(Real time,
                         const int lev,
                         FArrayBox& vel_fab,
                         const Box& domain) const
{
  // Must do this because probtype is a member of incflo so can not be "just passed" to the GPU
  auto lprobtype = probtype;

  IntVect dom_lo(domain.loVect());
  IntVect dom_hi(domain.hiVect());

  Array4<Real> const& vel_arr = vel_fab.array();

  IntVect vel_lo(vel_fab.loVect());
  IntVect vel_hi(vel_fab.hiVect());

  Array4<const int> const& bct_ilo = bc_ilo[lev]->array();
  Array4<const int> const& bct_ihi = bc_ihi[lev]->array();
  Array4<const int> const& bct_jlo = bc_jlo[lev]->array();
  Array4<const int> const& bct_jhi = bc_jhi[lev]->array();
  Array4<const int> const& bct_klo = bc_klo[lev]->array();
  Array4<const int> const& bct_khi = bc_khi[lev]->array();

  const int nlft = std::max(0, dom_lo[0]-vel_lo[0]);
  const int nbot = std::max(0, dom_lo[1]-vel_lo[1]);
  const int ndwn = std::max(0, dom_lo[2]-vel_lo[2]);

  const int nrgt = std::max(0, vel_hi[0]-dom_hi[0]);
  const int ntop = std::max(0, vel_hi[1]-dom_hi[1]);
  const int nup  = std::max(0, vel_hi[2]-dom_hi[2]);

  // Create InVects for following 2D Boxes
  IntVect bx_yz_lo_lo_2D(vel_lo), bx_yz_lo_hi_2D(vel_hi);
  IntVect bx_yz_hi_lo_2D(vel_lo), bx_yz_hi_hi_2D(vel_hi);
  IntVect bx_xz_lo_lo_2D(vel_lo), bx_xz_lo_hi_2D(vel_hi);
  IntVect bx_xz_hi_lo_2D(vel_lo), bx_xz_hi_hi_2D(vel_hi);
  IntVect bx_xy_lo_lo_2D(vel_lo), bx_xy_lo_hi_2D(vel_hi);
  IntVect bx_xy_hi_lo_2D(vel_lo), bx_xy_hi_hi_2D(vel_hi);

  // Fix lo and hi limits
  bx_yz_lo_lo_2D[0] = dom_lo[0]-1;
  bx_yz_lo_hi_2D[0] = dom_lo[0]-1;
  bx_yz_hi_lo_2D[0] = dom_hi[0]+1;
  bx_yz_hi_hi_2D[0] = dom_hi[0]+1;

  bx_xz_lo_lo_2D[1] = dom_lo[1]-1;
  bx_xz_lo_hi_2D[1] = dom_lo[1]-1;
  bx_xz_hi_lo_2D[1] = dom_hi[1]+1;
  bx_xz_hi_hi_2D[1] = dom_hi[1]+1;

  bx_xy_lo_lo_2D[2] = dom_lo[2]-1;
  bx_xy_lo_hi_2D[2] = dom_lo[2]-1;
  bx_xy_hi_lo_2D[2] = dom_hi[2]+1;
  bx_xy_hi_hi_2D[2] = dom_hi[2]+1;

  // Create 2D boxes for CUDA loops
  const Box bx_yz_lo_2D(bx_yz_lo_lo_2D, bx_yz_lo_hi_2D);
  const Box bx_yz_hi_2D(bx_yz_hi_lo_2D, bx_yz_hi_hi_2D);

  const Box bx_xz_lo_2D(bx_xz_lo_lo_2D, bx_xz_lo_hi_2D);
  const Box bx_xz_hi_2D(bx_xz_hi_lo_2D, bx_xz_hi_hi_2D);

  const Box bx_xy_lo_2D(bx_xy_lo_lo_2D, bx_xy_lo_hi_2D);
  const Box bx_xy_hi_2D(bx_xy_hi_lo_2D, bx_xy_hi_hi_2D);

  // Create InVects for following 3D Boxes
  IntVect bx_yz_lo_hi_3D(vel_hi), bx_xz_lo_hi_3D(vel_hi), bx_xy_lo_hi_3D(vel_hi);
  IntVect bx_yz_hi_lo_3D(vel_lo), bx_xz_hi_lo_3D(vel_lo), bx_xy_hi_lo_3D(vel_lo);

  // Fix lo and hi limits
  bx_yz_lo_hi_3D[0] = dom_lo[0]-1;
  bx_yz_hi_lo_3D[0] = dom_hi[0]+1;

  bx_xz_lo_hi_3D[1] = dom_lo[1]-1;
  bx_xz_hi_lo_3D[1] = dom_hi[1]+1;

  bx_xy_lo_hi_3D[2] = dom_lo[2]-1;
  bx_xy_hi_lo_3D[2] = dom_hi[2]+1;

  // Create 3D boxes for CUDA loops
  const Box bx_yz_lo_3D(vel_lo, bx_yz_lo_hi_3D);
  const Box bx_yz_hi_3D(bx_yz_hi_lo_3D, vel_hi);

  const Box bx_xz_lo_3D(vel_lo, bx_xz_lo_hi_3D);
  const Box bx_xz_hi_3D(bx_xz_hi_lo_3D, vel_hi);

  const Box bx_xy_lo_3D(vel_lo, bx_xy_lo_hi_3D);
  const Box bx_xy_hi_3D(bx_xy_hi_lo_3D, vel_hi);

  for(unsigned i(1); i <= 6; ++i)
  {
    m_bc_u[i] = get_bc_u(i);
    m_bc_v[i] = get_bc_v(i);
    m_bc_w[i] = get_bc_w(i);

    m_bc_r[i] = get_bc_r(i);
    m_bc_t[i] = get_bc_t(i);
  }

  const int minf = bc_list.get_minf();
  const int pinf = bc_list.get_pinf();
  const int pout = bc_list.get_pout();
  const int  nsw = bc_list.get_nsw();

  const amrex::Real* p_bc_u = m_bc_u.data();
  const amrex::Real* p_bc_v = m_bc_v.data();
  const amrex::Real* p_bc_w = m_bc_w.data();

  if (nlft > 0)
  {
    amrex::ParallelFor(bx_yz_lo_3D, 3, 
      [bct_ilo,dom_lo,dom_hi,pinf,pout,minf,lprobtype,p_bc_u,vel_arr] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
      const int bcv = bct_ilo(dom_lo[0]-1,j,k,1);
      const int bct = bct_ilo(dom_lo[0]-1,j,k,0);

      if((bct == pinf) or (bct == pout)) {
        vel_arr(i,j,k,n) = vel_arr(dom_lo[0],j,k,n);

      } else if(bct == minf) {
        if (n == 0)
        {
           vel_arr(i,j,k,n) = p_bc_u[bcv];
           if (lprobtype == 31)
           {
               Real y = (j + 0.5) / (dom_hi[1] - dom_lo[1] + 1);
               vel_arr(i,j,k,n) =  6.0 * p_bc_u[bcv] * y * (1.0 - y);
           }
        }
        else
          vel_arr(i,j,k,n) = 0;
      }
    });
  }

  if (nrgt > 0)
  {
    amrex::ParallelFor(bx_yz_hi_3D, 3, 
      [bct_ihi,dom_lo,dom_hi,pinf,pout,minf,lprobtype,p_bc_u,vel_arr] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
      const int bcv = bct_ihi(dom_hi[0]+1,j,k,1);
      const int bct = bct_ihi(dom_hi[0]+1,j,k,0);

      if((bct == pinf) or (bct == pout)) {
        vel_arr(i,j,k,n) = vel_arr(dom_hi[0],j,k,n);
      } else if(bct == minf)
      {
        if(n == 0)
          vel_arr(i,j,k,n) = p_bc_u[bcv];
        else
          vel_arr(i,j,k,n) = 0;
      }
    });
  }

  if (nbot > 0)
  {
    amrex::ParallelFor(bx_xz_lo_3D, 3, 
      [bct_jlo,dom_lo,dom_hi,pinf,pout,minf,lprobtype,p_bc_v,vel_arr] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
      const int bcv = bct_jlo(i,dom_lo[1]-1,k,1);
      const int bct = bct_jlo(i,dom_lo[1]-1,k,0);

      if((bct == pinf) or (bct == pout)) {
        vel_arr(i,j,k,n) = vel_arr(i,dom_lo[1],k,n);
      } else if(bct == minf)
      {
        if(n == 1)
        {
           vel_arr(i,j,k,n) = p_bc_v[bcv];
           if (lprobtype == 32)
           {
               Real z = (k + 0.5) / (dom_hi[2] - dom_lo[2] + 1);
               vel_arr(i,j,k,n) =  6.0 * p_bc_v[bcv] * z * (1.0 - z);
           }
        }
        else
        {
          vel_arr(i,j,k,n) = 0;
        }
      }
    });
  }

  if (ntop > 0)
  {
    amrex::ParallelFor(bx_xz_hi_3D, 3, 
      [bct_jhi,dom_lo,dom_hi,pinf,pout,minf,lprobtype,p_bc_v,vel_arr] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
      const int bcv = bct_jhi(i,dom_hi[1]+1,k,1);
      const int bct = bct_jhi(i,dom_hi[1]+1,k,0);

      if((bct == pinf) or (bct == pout)) {
        vel_arr(i,j,k,n) = vel_arr(i,dom_hi[1],k,n);
      } else if(bct == minf)
      {
        if(n == 1)
          vel_arr(i,j,k,n) = p_bc_v[bcv];
        else
          vel_arr(i,j,k,n) = 0;
      }
    });
  }

  if (ndwn > 0)
  {
    amrex::ParallelFor(bx_xy_lo_3D, 3, 
      [bct_klo,dom_lo,dom_hi,pinf,pout,minf,lprobtype,p_bc_w,vel_arr] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
      const int bcv = bct_klo(i,j,dom_lo[2]-1,1);
      const int bct = bct_klo(i,j,dom_lo[2]-1,0);

      if((bct == pinf) or (bct == pout)) {
        vel_arr(i,j,k,n) = vel_arr(i,j,dom_lo[2],n);
      } else if(bct == minf)
      {
        if(n == 2)
        {
           vel_arr(i,j,k,n) = p_bc_w[bcv];
           if (lprobtype == 33)
           {
               Real x = (i + 0.5) / (dom_hi[0] - dom_lo[0] + 1);
               vel_arr(i,j,k,n) =  6.0 * p_bc_w[bcv] * x * (1.0 - x);
           }
        }
        else
          vel_arr(i,j,k,n) = 0;
      }
    });
  }

  if (nup > 0)
  {
    amrex::ParallelFor(bx_xy_hi_3D, 3, 
      [bct_khi,dom_lo,dom_hi,pinf,pout,minf,lprobtype,p_bc_w,vel_arr] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
      const int bcv = bct_khi(i,j,dom_hi[2]+1,1);
      const int bct = bct_khi(i,j,dom_hi[2]+1,0);

      if((bct == pinf) or (bct == pout)) {
        vel_arr(i,j,k,n) = vel_arr(i,j,dom_hi[2],n);
      } else if(bct == minf)
      {
        if(n == 2)
          vel_arr(i,j,k,n) = p_bc_w[bcv];
        else
          vel_arr(i,j,k,n) = 0;
      }
    });
  }

/* *******************************************************************
   Do this section next to make sure nsw over-rides any previous minf
   ****************************************************************** */

  if (nlft > 0)
  {
    amrex::ParallelFor(bx_yz_lo_3D,
      [bct_ilo,dom_lo,nsw,p_bc_v,p_bc_w,vel_arr] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      const int bcv = bct_ilo(dom_lo[0]-1,j,k,1);
      const int bct = bct_ilo(dom_lo[0]-1,j,k,0);

      if(bct == nsw) {
        vel_arr(i,j,k,0) = 0.;
        vel_arr(i,j,k,1) = p_bc_v[bcv];
        vel_arr(i,j,k,2) = p_bc_w[bcv];
      }
    });
  }

  if (nrgt > 0)
  {
    amrex::ParallelFor(bx_yz_hi_3D,
      [bct_ihi,dom_hi,nsw,p_bc_v,p_bc_w,vel_arr] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      const int bcv = bct_ihi(dom_hi[0]+1,j,k,1);
      const int bct = bct_ihi(dom_hi[0]+1,j,k,0);

      if(bct == nsw) {
        vel_arr(i,j,k,0) = 0.;
        vel_arr(i,j,k,1) = p_bc_v[bcv];
        vel_arr(i,j,k,2) = p_bc_w[bcv];
      }
    });
  }

  if (nbot > 0)
  {
    amrex::ParallelFor(bx_xz_lo_3D,
      [bct_jlo,dom_lo,nsw,p_bc_u,p_bc_w,vel_arr] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      const int bcv = bct_jlo(i,dom_lo[1]-1,k,1);
      const int bct = bct_jlo(i,dom_lo[1]-1,k,0);

      if (bct == nsw) {
        vel_arr(i,j,k,0) = p_bc_u[bcv];
        vel_arr(i,j,k,1) = 0.;
        vel_arr(i,j,k,2) = p_bc_w[bcv];
      }
    });
  }

  if (ntop > 0)
  {
    amrex::ParallelFor(bx_xz_hi_3D,
      [bct_jhi,dom_hi,nsw,p_bc_u,p_bc_w,vel_arr] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      const int bcv = bct_jhi(i,dom_hi[1]+1,k,1);
      const int bct = bct_jhi(i,dom_hi[1]+1,k,0);

      if (bct == nsw) {
        vel_arr(i,j,k,0) = p_bc_u[bcv];
        vel_arr(i,j,k,1) = 0.;
        vel_arr(i,j,k,2) = p_bc_w[bcv];
      }
    });
  }

  if (ndwn > 0)
  {
    amrex::ParallelFor(bx_xy_lo_3D,
      [bct_klo,dom_lo,nsw,p_bc_u,p_bc_v,vel_arr] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      const int bcv = bct_klo(i,j,dom_lo[2]-1,1);
      const int bct = bct_klo(i,j,dom_lo[2]-1,0);

      if (bct == nsw) {
        vel_arr(i,j,k,0) = p_bc_u[bcv];
        vel_arr(i,j,k,1) = p_bc_v[bcv];
        vel_arr(i,j,k,2) = 0.;
      }
    });
  }

  if (nup > 0)
  {
    amrex::ParallelFor(bx_xy_hi_3D,
      [bct_khi,dom_hi,nsw,p_bc_u,p_bc_v,vel_arr] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      const int bcv = bct_khi(i,j,dom_hi[2]+1,1);
      const int bct = bct_khi(i,j,dom_hi[2]+1,0);

      if (bct == nsw) {
        vel_arr(i,j,k,0) = p_bc_u[bcv];
        vel_arr(i,j,k,1) = p_bc_v[bcv];
        vel_arr(i,j,k,2) = 0.;
      }
    });
  }
}
