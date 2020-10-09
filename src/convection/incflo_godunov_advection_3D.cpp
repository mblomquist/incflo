#include <incflo_godunov_plm.H>
#include <incflo_godunov_ppm.H>

#include <Godunov.H>

#ifdef AMREX_USE_EB
#include <AMReX_EB_slopes_K.H>
#endif

using namespace amrex;

namespace {
    std::pair<bool,bool> has_extdir_or_ho (BCRec const* bcrec, int ncomp, int dir)
    {   
        std::pair<bool,bool> r{false,false};
        for (int n = 0; n < ncomp; ++n) {
            r.first = r.first
                 or (bcrec[n].lo(dir) == BCType::ext_dir)
                 or (bcrec[n].lo(dir) == BCType::hoextrap);
            r.second = r.second
                 or (bcrec[n].hi(dir) == BCType::ext_dir)
                 or (bcrec[n].hi(dir) == BCType::hoextrap);
        }
        return r;
    }
}

void
godunov::compute_godunov_advection (int lev, Box const& bx, int ncomp,
                                    Array4<Real> const& dqdt,
                                    Array4<Real const> const& q,
                                    Array4<Real const> const& umac,
                                    Array4<Real const> const& vmac,
                                    Array4<Real const> const& wmac,
                                    Array4<Real const> const& fq,
                                    Vector<amrex::Geometry> geom,
                                    Real l_dt,
                                    BCRec const* pbc, int const* iconserv,
                                    Real* p, bool use_ppm, bool is_velocity )
{
    Box const& xbx = amrex::surroundingNodes(bx,0);
    Box const& ybx = amrex::surroundingNodes(bx,1);
    Box const& zbx = amrex::surroundingNodes(bx,2);
    Box const& bxg1 = amrex::grow(bx,1);
    Box xebox = Box(xbx).grow(1,1).grow(2,1);
    Box yebox = Box(ybx).grow(0,1).grow(2,1);
    Box zebox = Box(zbx).grow(0,1).grow(1,1);

    const Real dx = geom[lev].CellSize(0);
    const Real dy = geom[lev].CellSize(1);
    const Real dz = geom[lev].CellSize(2);
    Real dtdx = l_dt/dx;
    Real dtdy = l_dt/dy;
    Real dtdz = l_dt/dz;

    Box const& domain = geom[lev].Domain();
    const auto dlo = amrex::lbound(domain);
    const auto dhi = amrex::ubound(domain);
    const auto dxinv = geom[lev].InvCellSizeArray();

    Array4<Real> Imx = makeArray4(p, bxg1, ncomp);
    p +=         Imx.size();
    Array4<Real> Ipx = makeArray4(p, bxg1, ncomp);
    p +=         Ipx.size();
    Array4<Real> Imy = makeArray4(p, bxg1, ncomp);
    p +=         Imy.size();
    Array4<Real> Ipy = makeArray4(p, bxg1, ncomp);
    p +=         Ipy.size();
    Array4<Real> Imz = makeArray4(p, bxg1, ncomp);
    p +=         Imz.size();
    Array4<Real> Ipz = makeArray4(p, bxg1, ncomp);
    p +=         Ipz.size();
    Array4<Real> xlo = makeArray4(p, xebox, ncomp);
    p +=         xlo.size();
    Array4<Real> xhi = makeArray4(p, xebox, ncomp);
    p +=         xhi.size();
    Array4<Real> ylo = makeArray4(p, yebox, ncomp);
    p +=         ylo.size();
    Array4<Real> yhi = makeArray4(p, yebox, ncomp);
    p +=         yhi.size();
    Array4<Real> zlo = makeArray4(p, zebox, ncomp);
    p +=         zlo.size();
    Array4<Real> zhi = makeArray4(p, zebox, ncomp);
    p +=         zhi.size();
    Array4<Real> divu = makeArray4(p, bxg1, 1);
    p +=         divu.size();
    Array4<Real> xyzlo = makeArray4(p, bxg1, ncomp);
    p +=         xyzlo.size();
    Array4<Real> xyzhi = makeArray4(p, bxg1, ncomp);
    p +=         xyzhi.size();

    // Use PPM to generate Im and Ip */
    if (use_ppm) {
        amrex::ParallelFor(bxg1, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Godunov_ppm_fpu_x(i, j, k, n, l_dt, dx, Imx(i,j,k,n), Ipx(i,j,k,n),
                              q, umac, pbc[n], dlo.x, dhi.x);
            Godunov_ppm_fpu_y(i, j, k, n, l_dt, dy, Imy(i,j,k,n), Ipy(i,j,k,n),
                              q, vmac, pbc[n], dlo.y, dhi.y);
            Godunov_ppm_fpu_z(i, j, k, n, l_dt, dz, Imz(i,j,k,n), Ipz(i,j,k,n),
                              q, wmac, pbc[n], dlo.z, dhi.z);
        });

    // Use PLM to generate Im and Ip */
    } else {

        amrex::ParallelFor(xebox, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Godunov_plm_fpu_x(i, j, k, n, l_dt, dx, Imx(i,j,k,n), Ipx(i-1,j,k,n),
                              q, umac(i,j,k), pbc[n], dlo.x, dhi.x, is_velocity);
        });

        amrex::ParallelFor(yebox, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Godunov_plm_fpu_y(i, j, k, n, l_dt, dy, Imy(i,j,k,n), Ipy(i,j-1,k,n),
                              q, vmac(i,j,k), pbc[n], dlo.y, dhi.y, is_velocity);
        });

        amrex::ParallelFor(zebox, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Godunov_plm_fpu_z(i, j, k, n, l_dt, dz, Imz(i,j,k,n), Ipz(i,j,k-1,n),
                              q, wmac(i,j,k), pbc[n], dlo.z, dhi.z, is_velocity);
        });
    }

    amrex::ParallelFor(Box(divu), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        divu(i,j,k) = 0.0;
    });

    amrex::ParallelFor(
        xebox, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real uad = umac(i,j,k);
            Real fux = (amrex::Math::abs(uad) < small_vel)? 0. : 1.;
            bool uval = uad >= 0.;
            Real cons1 = (iconserv[n]) ? -0.5*l_dt*q(i-1,j,k,n)*divu(i-1,j,k) : 0.;
            Real cons2 = (iconserv[n]) ? -0.5*l_dt*q(i  ,j,k,n)*divu(i  ,j,k) : 0.;
            Real lo = Ipx(i-1,j,k,n) + cons1;
            Real hi = Imx(i  ,j,k,n) + cons2;
            if (fq) {
                lo += 0.5*l_dt*fq(i-1,j,k,n);
                hi += 0.5*l_dt*fq(i  ,j,k,n);
            }

            auto bc = pbc[n];

            xlo(i,j,k,n) = lo;
            Godunov_trans_xbc(i, j, k, n, q, lo, hi, uad, bc.lo(0), bc.hi(0), dlo.x, dhi.x, is_velocity);
            xhi(i,j,k,n) = hi;
            Real st = (uval) ? lo : hi;
            Imx(i,j,k,n) = fux*st + (1. - fux)*0.5*(hi + lo);

        },
        yebox, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real vad = vmac(i,j,k);
            Real fuy = (amrex::Math::abs(vad) < small_vel)? 0. : 1.;
            bool vval = vad >= 0.;
            Real cons1 = (iconserv[n]) ? -0.5*l_dt*q(i,j-1,k,n)*divu(i,j-1,k) : 0.;
            Real cons2 = (iconserv[n]) ? -0.5*l_dt*q(i,j  ,k,n)*divu(i,j  ,k) : 0.;
            Real lo = Ipy(i,j-1,k,n) + cons1;
            Real hi = Imy(i,j  ,k,n) + cons2;
            if (fq) {
                lo += 0.5*l_dt*fq(i,j-1,k,n);
                hi += 0.5*l_dt*fq(i,j  ,k,n);
            }

            auto bc = pbc[n];

            Godunov_trans_ybc(i, j, k, n, q, lo, hi, vad, bc.lo(1), bc.hi(1), dlo.y, dhi.y, is_velocity);

            ylo(i,j,k,n) = lo;
            yhi(i,j,k,n) = hi;
            Real st = (vval) ? lo : hi;
            Imy(i,j,k,n) = fuy*st + (1. - fuy)*0.5*(hi + lo);
        },
        zebox, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {

            Real wad = wmac(i,j,k);
            Real fuz = (amrex::Math::abs(wad) < small_vel) ? 0. : 1.;
            bool wval = wad >= 0.;
            auto bc = pbc[n];
            Real cons1 = (iconserv[n]) ? -0.5*l_dt*q(i,j,k-1,n)*divu(i,j,k-1) : 0.;
            Real cons2 = (iconserv[n]) ? -0.5*l_dt*q(i,j,k  ,n)*divu(i,j,k  ) : 0.;
            Real lo = Ipz(i,j,k-1,n) + cons1;
            Real hi = Imz(i,j,k  ,n) + cons2;
            if (fq) {
                lo += 0.5*l_dt*fq(i,j,k-1,n);
                hi += 0.5*l_dt*fq(i,j,k  ,n);
            }

            Godunov_trans_zbc(i, j, k, n, q, lo, hi, wad, bc.lo(2), bc.hi(2), dlo.z, dhi.z, is_velocity);

            zlo(i,j,k,n) = lo;
            zhi(i,j,k,n) = hi;
            Real st = (wval) ? lo : hi;
            Imz(i,j,k,n) = fuz*st + (1. - fuz)*0.5*(hi + lo);
        });

    Array4<Real> xedge = Imx;
    Array4<Real> yedge = Imy;
    Array4<Real> zedge = Imz;

    // We can reuse the space in Ipx, Ipy and Ipz.

    //
    // x-direction
    //
    Box const& xbxtmp = amrex::grow(bx,0,1);
    Array4<Real> yzlo = makeArray4(xyzlo.dataPtr(), amrex::surroundingNodes(xbxtmp,1), ncomp);
    Array4<Real> zylo = makeArray4(xyzhi.dataPtr(), amrex::surroundingNodes(xbxtmp,2), ncomp);
    amrex::ParallelFor(
    Box(zylo), ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        const auto bc = pbc[n];
        Real l_zylo, l_zyhi;
        Godunov_corner_couple_zy(l_zylo, l_zyhi,
                                 i, j, k, n, l_dt, dy, iconserv[n],
                                 zlo(i,j,k,n), zhi(i,j,k,n),
                                 q, divu, vmac, yedge);

        Real wad = wmac(i,j,k);
        Godunov_trans_zbc(i, j, k, n, q, l_zylo, l_zyhi, wad, bc.lo(2), bc.hi(2), dlo.z, dhi.z, is_velocity);

        Real st = (wad >= 0.) ? l_zylo : l_zyhi;
        Real fu = (amrex::Math::abs(wad) < small_vel) ? 0.0 : 1.0;
        zylo(i,j,k,n) = fu*st + (1.0 - fu) * 0.5 * (l_zyhi + l_zylo);
    },
    Box(yzlo), ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        const auto bc = pbc[n];
        Real l_yzlo, l_yzhi;
        Godunov_corner_couple_yz(l_yzlo, l_yzhi,
                                 i, j, k, n, l_dt, dz, iconserv[n],
                                 ylo(i,j,k,n), yhi(i,j,k,n),
                                 q, divu, wmac, zedge);

        Real vad = vmac(i,j,k);
        Godunov_trans_ybc(i, j, k, n, q, l_yzlo, l_yzhi, vad, bc.lo(1), bc.hi(1), dlo.y, dhi.y, is_velocity);

        Real st = (vad >= 0.) ? l_yzlo : l_yzhi;
        Real fu = (amrex::Math::abs(vad) < small_vel) ? 0.0 : 1.0;
        yzlo(i,j,k,n) = fu*st + (1.0 - fu) * 0.5 * (l_yzhi + l_yzlo);
    });
    //
    Array4<Real> qx = makeArray4(Ipx.dataPtr(), xbx, ncomp);
    amrex::ParallelFor(xbx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        Real stl, sth;

        if (iconserv[n]) {
            stl = xlo(i,j,k,n) - (0.5*dtdy)*(yzlo(i-1,j+1,k  ,n)*vmac(i-1,j+1,k  )
                                           - yzlo(i-1,j  ,k  ,n)*vmac(i-1,j  ,k  ))
                               - (0.5*dtdz)*(zylo(i-1,j  ,k+1,n)*wmac(i-1,j  ,k+1)
                                           - zylo(i-1,j  ,k  ,n)*wmac(i-1,j  ,k  ))
                + (0.5*dtdy)*q(i-1,j,k,n)*(vmac(i-1,j+1,k  ) - vmac(i-1,j,k))
                + (0.5*dtdz)*q(i-1,j,k,n)*(wmac(i-1,j  ,k+1) - wmac(i-1,j,k));

            sth = xhi(i,j,k,n) - (0.5*dtdy)*(yzlo(i,j+1,k  ,n)*vmac(i,j+1,k  )
                                           - yzlo(i,j  ,k  ,n)*vmac(i,j  ,k  ))
                               - (0.5*dtdz)*(zylo(i,j  ,k+1,n)*wmac(i,j  ,k+1)
                                           - zylo(i,j  ,k  ,n)*wmac(i,j  ,k  ))
                + (0.5*dtdy)*q(i,j,k,n)*(vmac(i,j+1,k  ) - vmac(i,j,k))
                + (0.5*dtdz)*q(i,j,k,n)*(wmac(i,j  ,k+1) - wmac(i,j,k));
        } else {
            stl = xlo(i,j,k,n) - (0.25*dtdy)*(vmac(i-1,j+1,k  ) + vmac(i-1,j,k)) *
                                             (yzlo(i-1,j+1,k,n) - yzlo(i-1,j,k,n))
                               - (0.25*dtdz)*(wmac(i-1,j,k+1  ) + wmac(i-1,j,k))*
                                             (zylo(i-1,j,k+1,n) - zylo(i-1,j,k,n));

            sth = xhi(i,j,k,n) - (0.25*dtdy)*(vmac(i,j+1,k  ) + vmac(i,j,k))*
                                             (yzlo(i,j+1,k,n) - yzlo(i,j,k,n))
                               - (0.25*dtdz)*(wmac(i,j,k+1  ) + wmac(i,j,k))*
                                             (zylo(i,j,k+1,n) - zylo(i,j,k,n));
        }

        auto bc = pbc[n];
        Godunov_cc_xbc_lo(i, j, k, n, q, stl, sth, umac, bc.lo(0), dlo.x, is_velocity);
        Godunov_cc_xbc_hi(i, j, k, n, q, stl, sth, umac, bc.hi(0), dhi.x, is_velocity);

        Real temp = (umac(i,j,k) >= 0.) ? stl : sth;
        temp = (amrex::Math::abs(umac(i,j,k)) < small_vel) ? 0.5*(stl + sth) : temp;
        qx(i,j,k,n) = temp;
    });

    //
    // y-direction
    //
    Box const& ybxtmp = amrex::grow(bx,1,1);
    Array4<Real> xzlo = makeArray4(xyzlo.dataPtr(), amrex::surroundingNodes(ybxtmp,0), ncomp);
    Array4<Real> zxlo = makeArray4(xyzhi.dataPtr(), amrex::surroundingNodes(ybxtmp,2), ncomp);
    amrex::ParallelFor(
    Box(xzlo), ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        const auto bc = pbc[n];
        Real l_xzlo, l_xzhi;
        Godunov_corner_couple_xz(l_xzlo, l_xzhi,
                                 i, j, k, n, l_dt, dz, iconserv[n],
                                 xlo(i,j,k,n),  xhi(i,j,k,n),
                                 q, divu, wmac, zedge);

        Real uad = umac(i,j,k);
        Godunov_trans_xbc(i, j, k, n, q, l_xzlo, l_xzhi, uad, bc.lo(0), bc.hi(0), dlo.x, dhi.x, is_velocity);

        Real st = (uad >= 0.) ? l_xzlo : l_xzhi;
        Real fu = (amrex::Math::abs(uad) < small_vel) ? 0.0 : 1.0;
        xzlo(i,j,k,n) = fu*st + (1.0 - fu) * 0.5 * (l_xzhi + l_xzlo);
    },
    Box(zxlo), ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        const auto bc = pbc[n];
        Real l_zxlo, l_zxhi;
        Godunov_corner_couple_zx(l_zxlo, l_zxhi,
                                 i, j, k, n, l_dt, dx, iconserv[n],
                                 zlo(i,j,k,n), zhi(i,j,k,n),
                                 q, divu, umac, xedge);

        Real wad = wmac(i,j,k);
        Godunov_trans_zbc(i, j, k, n, q, l_zxlo, l_zxhi, wad, bc.lo(2), bc.hi(2), dlo.z, dhi.z, is_velocity);

        Real st = (wad >= 0.) ? l_zxlo : l_zxhi;
        Real fu = (amrex::Math::abs(wad) < small_vel) ? 0.0 : 1.0;
        zxlo(i,j,k,n) = fu*st + (1.0 - fu) * 0.5 * (l_zxhi + l_zxlo);
    });
    //

    Array4<Real> qy = makeArray4(Ipy.dataPtr(), ybx, ncomp);
    amrex::ParallelFor(ybx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        Real stl, sth;

        if (iconserv[n]){
            stl = ylo(i,j,k,n) - (0.5*dtdx)*(xzlo(i+1,j-1,k  ,n)*umac(i+1,j-1,k  )
                                           - xzlo(i  ,j-1,k  ,n)*umac(i  ,j-1,k  ))
                               - (0.5*dtdz)*(zxlo(i  ,j-1,k+1,n)*wmac(i  ,j-1,k+1)
                                           - zxlo(i  ,j-1,k  ,n)*wmac(i  ,j-1,k  ))
                + (0.5*dtdx)*q(i,j-1,k,n)*(umac(i+1,j-1,k  ) - umac(i,j-1,k))
                + (0.5*dtdz)*q(i,j-1,k,n)*(wmac(i  ,j-1,k+1) - wmac(i,j-1,k));

            sth = yhi(i,j,k,n) - (0.5*dtdx)*(xzlo(i+1,j,k  ,n)*umac(i+1,j,k  )
                                           - xzlo(i  ,j,k  ,n)*umac(i  ,j,k  ))
                               - (0.5*dtdz)*(zxlo(i  ,j,k+1,n)*wmac(i  ,j,k+1)
                                           - zxlo(i  ,j,k  ,n)*wmac(i  ,j,k  ))
                + (0.5*dtdx)*q(i,j,k,n)*(umac(i+1,j,k  ) - umac(i,j,k))
                + (0.5*dtdz)*q(i,j,k,n)*(wmac(i  ,j,k+1) - wmac(i,j,k));
        } else {
            stl = ylo(i,j,k,n) - (0.25*dtdx)*(umac(i+1,j-1,k    ) + umac(i,j-1,k))*
                                             (xzlo(i+1,j-1,k  ,n) - xzlo(i,j-1,k,n))
                               - (0.25*dtdz)*(wmac(i  ,j-1,k+1  ) + wmac(i,j-1,k))*
                                             (zxlo(i  ,j-1,k+1,n) - zxlo(i,j-1,k,n));

            sth = yhi(i,j,k,n) - (0.25*dtdx)*(umac(i+1,j,k  ) + umac(i,j,k))*
                                             (xzlo(i+1,j,k,n) - xzlo(i,j,k,n))
                               - (0.25*dtdz)*(wmac(i,j,k+1  ) + wmac(i,j,k))*
                                             (zxlo(i,j,k+1,n) - zxlo(i,j,k,n));
        }

        auto bc = pbc[n];
        Godunov_cc_ybc_lo(i, j, k, n, q, stl, sth, vmac, bc.lo(1), dlo.y, is_velocity);
        Godunov_cc_ybc_hi(i, j, k, n, q, stl, sth, vmac, bc.hi(1), dhi.y, is_velocity);

        Real temp = (vmac(i,j,k) >= 0.) ? stl : sth;
        temp = (amrex::Math::abs(vmac(i,j,k)) < small_vel) ? 0.5*(stl + sth) : temp;
        qy(i,j,k,n) = temp;
    });

    //
    // z-direcion
    //
    Box const& zbxtmp = amrex::grow(bx,2,1);
    Array4<Real> xylo = makeArray4(xyzlo.dataPtr(), amrex::surroundingNodes(zbxtmp,0), ncomp);
    Array4<Real> yxlo = makeArray4(xyzhi.dataPtr(), amrex::surroundingNodes(zbxtmp,1), ncomp);
    amrex::ParallelFor(
    Box(xylo), ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        const auto bc = pbc[n];
        Real l_xylo, l_xyhi;
        Godunov_corner_couple_xy(l_xylo, l_xyhi,
                                 i, j, k, n, l_dt, dy, iconserv[n],
                                 xlo(i,j,k,n), xhi(i,j,k,n),
                                 q, divu, vmac, yedge);

        Real uad = umac(i,j,k);
        Godunov_trans_xbc(i, j, k, n, q, l_xylo, l_xyhi, uad, bc.lo(0), bc.hi(0), dlo.x, dhi.x, is_velocity);

        Real st = (uad >= 0.) ? l_xylo : l_xyhi;
        Real fu = (amrex::Math::abs(uad) < small_vel) ? 0.0 : 1.0;
        xylo(i,j,k,n) = fu*st + (1.0 - fu) * 0.5 * (l_xyhi + l_xylo);
    },
    Box(yxlo), ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        const auto bc = pbc[n];
        Real l_yxlo, l_yxhi;
        Godunov_corner_couple_yx(l_yxlo, l_yxhi,
                                 i, j, k, n, l_dt, dx, iconserv[n],
                                 ylo(i,j,k,n), yhi(i,j,k,n),
                                 q, divu, umac, xedge);

        Real vad = vmac(i,j,k);
        Godunov_trans_ybc(i, j, k, n, q, l_yxlo, l_yxhi, vad, bc.lo(1), bc.hi(1), dlo.y, dhi.y, is_velocity);

        Real st = (vad >= 0.) ? l_yxlo : l_yxhi;
        Real fu = (amrex::Math::abs(vad) < small_vel) ? 0.0 : 1.0;
        yxlo(i,j,k,n) = fu*st + (1.0 - fu) * 0.5 * (l_yxhi + l_yxlo);
    });
    //
    Array4<Real> qz = makeArray4(Ipz.dataPtr(), zbx, ncomp);
    amrex::ParallelFor(zbx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        Real stl, sth;

        if (iconserv[n]) {
            stl = zlo(i,j,k,n) - (0.5*dtdx)*(xylo(i+1,j  ,k-1,n)*umac(i+1,j  ,k-1)
                                           - xylo(i  ,j  ,k-1,n)*umac(i  ,j  ,k-1))
                               - (0.5*dtdy)*(yxlo(i  ,j+1,k-1,n)*vmac(i  ,j+1,k-1)
                                           - yxlo(i  ,j  ,k-1,n)*vmac(i  ,j  ,k-1))
                + (0.5*dtdx)*q(i,j,k-1,n)*(umac(i+1,j,k-1) -umac(i,j,k-1))
                + (0.5*dtdy)*q(i,j,k-1,n)*(vmac(i,j+1,k-1) -vmac(i,j,k-1));

            sth = zhi(i,j,k,n) - (0.5*dtdx)*(xylo(i+1,j  ,k,n)*umac(i+1,j  ,k)
                                           - xylo(i  ,j  ,k,n)*umac(i  ,j  ,k))
                               - (0.5*dtdy)*(yxlo(i  ,j+1,k,n)*vmac(i  ,j+1,k)
                                           - yxlo(i  ,j  ,k,n)*vmac(i  ,j  ,k))
                + (0.5*dtdx)*q(i,j,k,n)*(umac(i+1,j,k) -umac(i,j,k))
                + (0.5*dtdy)*q(i,j,k,n)*(vmac(i,j+1,k) -vmac(i,j,k));
        } else {
            stl = zlo(i,j,k,n) - (0.25*dtdx)*(umac(i+1,j  ,k-1  ) + umac(i,j,k-1))*
                                             (xylo(i+1,j  ,k-1,n) - xylo(i,j,k-1,n))
                               - (0.25*dtdy)*(vmac(i  ,j+1,k-1  ) + vmac(i,j,k-1))*
                                             (yxlo(i  ,j+1,k-1,n) - yxlo(i,j,k-1,n));

            sth = zhi(i,j,k,n) - (0.25*dtdx)*(umac(i+1,j  ,k  ) + umac(i,j,k))*
                                             (xylo(i+1,j  ,k,n) - xylo(i,j,k,n))
                               - (0.25*dtdy)*(vmac(i  ,j+1,k  ) + vmac(i,j,k))*
                                             (yxlo(i  ,j+1,k,n) - yxlo(i,j,k,n));
        }

        auto bc = pbc[n];
        Godunov_cc_zbc_lo(i, j, k, n, q, stl, sth, wmac, bc.lo(2),  dlo.z, is_velocity);
        Godunov_cc_zbc_hi(i, j, k, n, q, stl, sth, wmac, bc.hi(2),  dhi.z, is_velocity);

        Real temp = (wmac(i,j,k) >= 0.) ? stl : sth;
        temp = (amrex::Math::abs(wmac(i,j,k)) < small_vel) ? 0.5*(stl + sth) : temp;
        qz(i,j,k,n) = temp;
    });

    amrex::ParallelFor(bx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        if (iconserv[n])
        {
            dqdt(i,j,k,n) = dxinv[0]*( umac(i  ,j,k)*qx(i  ,j,k,n) -
                                       umac(i+1,j,k)*qx(i+1,j,k,n) )
                +           dxinv[1]*( vmac(i,j  ,k)*qy(i,j  ,k,n) -
                                       vmac(i,j+1,k)*qy(i,j+1,k,n))
                +           dxinv[2]*( wmac(i,j,k  )*qz(i,j,k  ,n) -
                                       wmac(i,j,k+1)*qz(i,j,k+1,n) );
        } else {
            dqdt(i,j,k,n) = 0.5*dxinv[0]*(umac(i,j,k  ) + umac(i+1,j  ,k  ))
                *                        (qx  (i,j,k,n) - qx  (i+1,j  ,k  ,n))
                +           0.5*dxinv[1]*(vmac(i,j,k  ) + vmac(i  ,j+1,k  ))
                *                        (qy  (i,j,k,n) - qy  (i  ,j+1,k  ,n))
                +           0.5*dxinv[2]*(wmac(i,j,k  ) + wmac(i  ,j  ,k+1))
                *                        (qz  (i,j,k,n) - qz  (i  ,j  ,k+1,n));
       }
    });
}

#ifdef AMREX_USE_EB
void
godunov::compute_godunov_advection_eb (int lev, Box const& bx, int ncomp,
                                       AMREX_D_DECL(Array4<Real> const& fx,
                                                    Array4<Real> const& fy,
                                                    Array4<Real> const& fz),
                                       Array4<Real const> const& q,
                                       AMREX_D_DECL(Array4<Real const> const& umac,
                                                    Array4<Real const> const& vmac,
                                                    Array4<Real const> const& wmac),
                                       Array4<Real const> const& fq,
                                       BCRec const* h_bcrec,
                                       BCRec const* d_bcrec,
                                       Array4<EBCellFlag const> const& flag,
                                       AMREX_D_DECL(Array4<Real const> const& fcx,
                                                    Array4<Real const> const& fcy,
                                                    Array4<Real const> const& fcz),
                                       Array4<Real const> const& ccc,
                                       Vector<Geometry> geom,
                                       Array4<Real const> const& vel_c,
                                       Real m_dt)
{
    constexpr Real small_vel = 1.e-10;

    const auto dxinv = geom[lev].InvCellSizeArray();
    const Box& domain_box = geom[lev].Domain();
    AMREX_D_TERM(
        const int domain_ilo = domain_box.smallEnd(0);
        const int domain_ihi = domain_box.bigEnd(0);,
        const int domain_jlo = domain_box.smallEnd(1);
        const int domain_jhi = domain_box.bigEnd(1);,
        const int domain_klo = domain_box.smallEnd(2);
        const int domain_khi = domain_box.bigEnd(2););

    AMREX_D_TERM(Box const& xbx = amrex::surroundingNodes(bx,0);,
                 Box const& ybx = amrex::surroundingNodes(bx,1);,
                 Box const& zbx = amrex::surroundingNodes(bx,2););

    // ****************************************************************************
    // Decide whether the stencil at each cell might need to see values that
    //     live on face centroids rather than cell centroids, i.e.
    //     are at a domain boundary with ext_dir or hoextrap boundary conditions
    // ****************************************************************************

    auto extdir_lohi_x = has_extdir_or_ho(h_bcrec, ncomp, static_cast<int>(Direction::x));
    bool has_extdir_or_ho_lo_x = extdir_lohi_x.first;
    bool has_extdir_or_ho_hi_x = extdir_lohi_x.second;

    auto extdir_lohi_y = has_extdir_or_ho(h_bcrec, ncomp, static_cast<int>(Direction::y));
    bool has_extdir_or_ho_lo_y = extdir_lohi_y.first;
    bool has_extdir_or_ho_hi_y = extdir_lohi_y.second;

#if (AMREX_SPACEDIM == 3)
    auto extdir_lohi_z = has_extdir_or_ho(h_bcrec, ncomp, static_cast<int>(Direction::z));
    bool has_extdir_or_ho_lo_z = extdir_lohi_z.first;
    bool has_extdir_or_ho_hi_z = extdir_lohi_z.second;
#endif

    if ((has_extdir_or_ho_lo_x and domain_ilo >= xbx.smallEnd(0)-1) or
        (has_extdir_or_ho_hi_x and domain_ihi <= xbx.bigEnd(0)    ) or 
        (has_extdir_or_ho_lo_y and domain_jlo >= ybx.smallEnd(1)-1) or
        (has_extdir_or_ho_hi_y and domain_jhi <= ybx.bigEnd(1)    ) 
#if (AMREX_SPACEDIM == 2)
        )
#elif (AMREX_SPACEDIM == 3)
        or 
        (has_extdir_or_ho_lo_z and domain_jlo >= zbx.smallEnd(2)-1) or
        (has_extdir_or_ho_hi_z and domain_jhi <= zbx.bigEnd(2)    ) )
#endif
    {

        // ****************************************************************************
        // Predict to x-faces
        // ****************************************************************************
        amrex::ParallelFor(xbx, ncomp,
        [d_bcrec,q,ccc,flag,umac,small_vel,fx,
        AMREX_D_DECL(domain_ilo,domain_jlo,domain_klo),
        AMREX_D_DECL(domain_ihi,domain_jhi,domain_khi),
        AMREX_D_DECL(fcx,fcy,fcz),m_dt,fq,vel_c,dxinv]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {

           AMREX_D_TERM(bool extdir_or_ho_ilo = (d_bcrec[n].lo(0) == BCType::ext_dir) or
                                                (d_bcrec[n].lo(0) == BCType::hoextrap);,
                        bool extdir_or_ho_jlo = (d_bcrec[n].lo(1) == BCType::ext_dir) or
                                                (d_bcrec[n].lo(1) == BCType::hoextrap);,
                        bool extdir_or_ho_klo = (d_bcrec[n].lo(2) == BCType::ext_dir) or
                                                (d_bcrec[n].lo(2) == BCType::hoextrap););

           AMREX_D_TERM(bool extdir_or_ho_ihi = (d_bcrec[n].hi(0) == BCType::ext_dir) or
                                                (d_bcrec[n].hi(0) == BCType::hoextrap);,
                        bool extdir_or_ho_jhi = (d_bcrec[n].hi(1) == BCType::ext_dir) or
                                                (d_bcrec[n].hi(1) == BCType::hoextrap);,
                        bool extdir_or_ho_khi = (d_bcrec[n].hi(2) == BCType::ext_dir) or
                                                (d_bcrec[n].hi(2) == BCType::hoextrap););
           Real qs;

           if (flag(i,j,k).isConnected(-1,0,0)) 
           {
               if (i <= domain_ilo && (d_bcrec[n].lo(0) == BCType::ext_dir)) {
                   qs = q(domain_ilo-1,j,k,n);
               } else if (i >= domain_ihi+1 && (d_bcrec[n].hi(0) == BCType::ext_dir)) {
                   qs = q(domain_ihi+1,j,k,n);
               } else {

                   Real yf = fcx(i,j,k,0); // local (y,z) of centroid of z-face we are extrapolating to
#if (AMREX_SPACEDIM == 3)
                   Real zf = fcx(i,j,k,1);
#endif 
                   // Compute slopes of component "n" of q
                   const auto& slopes_eb_hi = amrex_calc_slopes_extdir_eb(i,j,k,n,q,ccc,
                                              AMREX_D_DECL(fcx,fcy,fcz), flag,
                                              AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo), 
                                              AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi), 
                                              AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo), 
                                              AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

                   AMREX_D_TERM(Real xc = ccc(i,j,k,0);, // centroid of cell (i,j,k)
                                Real yc = ccc(i,j,k,1);,
                                Real zc = ccc(i,j,k,2););
 
                   AMREX_D_TERM(Real delta_x = 0.5 + xc;,
                                Real delta_y = yf  - yc;,
                                Real delta_z = zf  - zc;);

                   //Adding temporal term with the normal derivative to the face
                   Real temp_u = -0.5*vel_c(i,j,k,0)*m_dt*dxinv[0];
 
#if (AMREX_SPACEDIM == 3) 
                   Real qpls = q(i  ,j,k,n) - (delta_x + temp_u) * slopes_eb_hi[0]
                                            + (delta_y         ) * slopes_eb_hi[1]
                                            + (delta_z         ) * slopes_eb_hi[2];
#else
                   Real qpls = q(i  ,j,k,n) - (delta_x + temp_u) * slopes_eb_hi[0]
                                            + (delta_y         ) * slopes_eb_hi[1];
#endif
                   Real cc_qmax = amrex::max(q(i,j,k,n),q(i-1,j,k,n));
                   Real cc_qmin = amrex::min(q(i,j,k,n),q(i-1,j,k,n));

                   qpls = amrex::max(amrex::min(qpls, cc_qmax), cc_qmin);

                   //Adding trans_force
                   if (fq) {
                       qpls += 0.5*m_dt*fq(i  ,j,k,n);
                   }
#if (AMREX_SPACEDIM == 3)
                   //Adding transverse derivative
                   qpls += - (0.5*m_dt)*(vel_c(i,j,k,1))*dxinv[1]*
                                              (slopes_eb_hi[1]);
                   qpls += - (0.5*m_dt)*(vel_c(i,j,k,2))*dxinv[2]*
                                              (slopes_eb_hi[2]);
#else
                   //Adding transverse derivative
                   qpls += - (0.5*m_dt)*(vel_c(i,j,k,1))*dxinv[1]*
                                              (slopes_eb_hi[1]);
#endif
    
                   AMREX_D_TERM(xc = ccc(i-1,j,k,0);, // centroid of cell (i-1,j,k)
                                yc = ccc(i-1,j,k,1);,
                                zc = ccc(i-1,j,k,2););
    
                   AMREX_D_TERM(delta_x = 0.5 - xc;,
                                delta_y = yf  - yc;,
                                delta_z = zf  - zc;);

                   // Compute slopes of component "n" of q
                   const auto& slopes_eb_lo = amrex_calc_slopes_extdir_eb(i-1,j,k,n,q,ccc,
                                              AMREX_D_DECL(fcx,fcy,fcz), flag,
                                              AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                              AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                              AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                              AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

                   //Adding temporal term with the normal derivative to the face
                   temp_u = -0.5*vel_c(i-1,j,k,0)*m_dt*dxinv[0];

#if (AMREX_SPACEDIM == 3)    
                   Real qmns = q(i-1,j,k,n) + (delta_x + temp_u) * slopes_eb_lo[0]
                                            + (delta_y         ) * slopes_eb_lo[1]
                                            + (delta_z         ) * slopes_eb_lo[2];
#else
                   Real qmns = q(i-1,j,k,n) + (delta_x + temp_u) * slopes_eb_lo[0]
                                            + (delta_y         ) * slopes_eb_lo[1];
#endif
                   qmns = amrex::max(amrex::min(qmns, cc_qmax), cc_qmin);

                   //Adding trans_force
                   if (fq) {
                       qmns += 0.5*m_dt*fq(i-1,j,k,n);
                   }

#if (AMREX_SPACEDIM == 3)
                   //Adding transverse derivative
                   qmns += - (0.5*m_dt)*(vel_c(i-1,j,k,1)) * dxinv[1]*
                                                   (slopes_eb_lo[1]);
                   qmns += - (0.5*m_dt)*(vel_c(i-1,j,k,2)) * dxinv[2]*
                                                   (slopes_eb_lo[2]);
#else
                   //Adding transverse derivative
                   qmns += - (0.5*m_dt)*(vel_c(i-1,j,k,1)) * dxinv[1]*
                                                   (slopes_eb_lo[1]);
#endif

                   if (umac(i,j,k) > small_vel) {
                       qs = qmns;
                   } else if (umac(i,j,k) < -small_vel) {
                       qs = qpls;
                   } else {
                       qs = 0.5*(qmns+qpls);
                   }
               }

               fx(i,j,k,n) = umac(i,j,k) * qs;
   
           } else {
               fx(i,j,k,n) = 0.0;
           }
        });

        // ****************************************************************************
        // Predict to y-faces
        // ****************************************************************************
        amrex::ParallelFor(ybx, ncomp,
        [d_bcrec,q,ccc,flag,vmac,small_vel,fy,
         AMREX_D_DECL(domain_ilo,domain_jlo,domain_klo),
         AMREX_D_DECL(domain_ihi,domain_jhi,domain_khi),
         AMREX_D_DECL(fcx,fcy,fcz),m_dt,fq,vel_c,dxinv]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real qs;
            if (flag(i,j,k).isConnected(0,-1,0)) 
            {
                AMREX_D_TERM(bool extdir_or_ho_ilo = (d_bcrec[n].lo(0) == BCType::ext_dir) or
                                                     (d_bcrec[n].lo(0) == BCType::hoextrap);,
                             bool extdir_or_ho_jlo = (d_bcrec[n].lo(1) == BCType::ext_dir) or
                                                     (d_bcrec[n].lo(1) == BCType::hoextrap);,
                             bool extdir_or_ho_klo = (d_bcrec[n].lo(2) == BCType::ext_dir) or
                                                     (d_bcrec[n].lo(2) == BCType::hoextrap););
                AMREX_D_TERM(bool extdir_or_ho_ihi = (d_bcrec[n].hi(0) == BCType::ext_dir) or
                                                     (d_bcrec[n].hi(0) == BCType::hoextrap);,
                             bool extdir_or_ho_jhi = (d_bcrec[n].hi(1) == BCType::ext_dir) or
                                                     (d_bcrec[n].hi(1) == BCType::hoextrap);,
                             bool extdir_or_ho_khi = (d_bcrec[n].hi(2) == BCType::ext_dir) or
                                                     (d_bcrec[n].hi(2) == BCType::hoextrap););

                   Real yf = fcx(i,j,k,0); // local (y,z) of centroid of z-face we are extrapolating to

                if (j <= domain_jlo && (d_bcrec[n].lo(1) == BCType::ext_dir)) {
                    qs = q(i,domain_jlo-1,k,n);
                } else if (j >= domain_jhi+1 && (d_bcrec[n].hi(1) == BCType::ext_dir)) {
                    qs = q(i,domain_jhi+1,k,n);
                } else {

                   Real yf = fcx(i,j,k,0); // local (y,z) of centroid of z-face we are extrapolating to

                   Real xf = fcy(i,j,k,0); // local (x,z) of centroid of z-face we are extrapolating to
#if (AMREX_SPACEDIM == 3)
                   Real zf = fcy(i,j,k,1);
#endif

                   AMREX_D_TERM(Real xc = ccc(i,j,k,0);, // centroid of cell (i,j,k)
                                Real yc = ccc(i,j,k,1);,
                                Real zc = ccc(i,j,k,2););
 
                   AMREX_D_TERM(Real delta_x = xf  - xc;,
                                Real delta_y = 0.5 + yc;,
                                Real delta_z = zf  - zc;);
    
                   Real cc_qmax = amrex::max(q(i,j,k,n),q(i,j-1,k,n));
                   Real cc_qmin = amrex::min(q(i,j,k,n),q(i,j-1,k,n));
     
                   // Compute slopes of component "n" of q
                   const auto& slopes_eb_hi = amrex_calc_slopes_extdir_eb(i,j,k,n,q,ccc,
                                              AMREX_D_DECL(fcx,fcy,fcz), flag,
                                              AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                              AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                              AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                              AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

                   //Adding temporal term with the normal derivative to the face
                   Real temp_v = -0.5*vel_c(i,j,k,1)*m_dt*dxinv[1]; 

#if (AMREX_SPACEDIM == 3)
                   Real qpls = q(i,j  ,k,n) + (delta_x         ) * slopes_eb_hi[0]
                                            - (delta_y + temp_v) * slopes_eb_hi[1]
                                            + (delta_z         ) * slopes_eb_hi[2];
#else
                   Real qpls = q(i,j  ,k,n) + (delta_x         ) * slopes_eb_hi[0]
                                            - (delta_y + temp_v) * slopes_eb_hi[1];
#endif
                   qpls = amrex::max(amrex::min(qpls, cc_qmax), cc_qmin);
    
                   //Adding trans_force
                   if (fq) {
                       qpls += 0.5*m_dt*fq(i,j  ,k,n);
                   }

#if (AMREX_SPACEDIM == 3)
                   //Adding transverse derivative
                   qpls += - (0.5*m_dt)*(vel_c(i,j,k,0))*dxinv[0]*
                                              (slopes_eb_hi[0]);
                   qpls += - (0.5*m_dt)*(vel_c(i,j,k,2))*dxinv[2]*
                                              (slopes_eb_hi[2]);
#else
                   //Adding transverse derivative
                   qpls += - (0.5*m_dt)*(vel_c(i,j,k,0))*dxinv[0]*
                                              (slopes_eb_hi[0]);
#endif

                   AMREX_D_TERM(xc = ccc(i,j-1,k,0);, // centroid of cell (i-1,j,k)
                                yc = ccc(i,j-1,k,1);,
                                zc = ccc(i,j-1,k,2););
    
                   AMREX_D_TERM(delta_x = xf  - xc;,
                                delta_y = 0.5 - yc;,
                                delta_z = zf  - zc;);

                   // Compute slopes of component "n" of q
                   const auto& slopes_eb_lo = amrex_calc_slopes_extdir_eb(i,j-1,k,n,q,ccc,
                                              AMREX_D_DECL(fcx,fcy,fcz), flag,
                                              AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                              AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                              AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                              AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

                   //Adding temporal term with the normal derivative to the face
                   temp_v = -0.5*vel_c(i,j-1,k,1)*m_dt*dxinv[1];

#if (AMREX_SPACEDIM == 3)    
                   Real qmns = q(i,j-1,k,n) + (delta_x         ) * slopes_eb_lo[0]
                                            + (delta_y + temp_v) * slopes_eb_lo[1]
                                            + (delta_z         ) * slopes_eb_lo[2];
#else
                   Real qmns = q(i,j-1,k,n) + (delta_x         ) * slopes_eb_lo[0]
                                            + (delta_y + temp_v) * slopes_eb_lo[1];
#endif
                   qmns = amrex::max(amrex::min(qmns, cc_qmax), cc_qmin);

                   //Adding trans_force
                   if (fq) {
                       qmns += 0.5*m_dt*fq(i,j-1,k,n);
                   }

#if (AMREX_SPACEDIM == 3)
                   //Adding transverse derivative
                   qmns += - (0.5*m_dt)*(vel_c(i,j-1,k,0))*dxinv[0]*
                                              (slopes_eb_lo[0]);
                   qmns += - (0.5*m_dt)*(vel_c(i,j-1,k,2))*dxinv[2]*
                                              (slopes_eb_lo[2]);
#else
                   //Adding transverse derivative
                   qmns += - (0.5*m_dt)*(vel_c(i,j-1,k,0))*dxinv[0]*
                                              (slopes_eb_lo[0]);
#endif

                    if (vmac(i,j,k) > small_vel) {
                        qs = qmns;
                    } else if (vmac(i,j,k) < -small_vel) {
                        qs = qpls;
                    } else {
                        qs = 0.5*(qmns+qpls);
                    }
                }

                fy(i,j,k,n) = vmac(i,j,k) * qs;

           } else {
                fy(i,j,k,n) = 0.0;
           }
        });

        // ****************************************************************************
        // Predict to z-faces
        // ****************************************************************************
#if (AMREX_SPACEDIM == 3)
        amrex::ParallelFor(zbx, ncomp,
        [d_bcrec,q,ccc,flag,wmac,small_vel,fz,
         domain_ilo,domain_jlo,domain_klo,
         domain_ihi,domain_jhi,domain_khi,
         fcx,fcy,fcz,m_dt,fq,vel_c,dxinv]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            if (flag(i,j,k).isConnected(0,0,-1)) {

                bool extdir_or_ho_ilo = (d_bcrec[n].lo(0) == BCType::ext_dir) or
                                        (d_bcrec[n].lo(0) == BCType::hoextrap);
                bool extdir_or_ho_ihi = (d_bcrec[n].hi(0) == BCType::ext_dir) or
                                        (d_bcrec[n].hi(0) == BCType::hoextrap);
                bool extdir_or_ho_jlo = (d_bcrec[n].lo(1) == BCType::ext_dir) or
                                        (d_bcrec[n].lo(1) == BCType::hoextrap);
                bool extdir_or_ho_jhi = (d_bcrec[n].hi(1) == BCType::ext_dir) or
                                        (d_bcrec[n].hi(1) == BCType::hoextrap);
                bool extdir_or_ho_klo = (d_bcrec[n].lo(2) == BCType::ext_dir) or
                                        (d_bcrec[n].lo(2) == BCType::hoextrap);
                bool extdir_or_ho_khi = (d_bcrec[n].hi(2) == BCType::ext_dir) or
                                        (d_bcrec[n].hi(2) == BCType::hoextrap);

                Real qs;
                if (k <= domain_klo && (d_bcrec[n].lo(2) == BCType::ext_dir)) {
                    qs = q(i,j,domain_klo-1,n);
                } else if (k >= domain_khi+1 && (d_bcrec[n].hi(2) == BCType::ext_dir)) {
                    qs = q(i,j,domain_khi+1,n);
                } else {

                    Real xf = fcz(i,j,k,0); // local (x,y) of centroid of z-face we are extrapolating to
                    Real yf = fcz(i,j,k,1);
 
                    Real xc = ccc(i,j,k,0); // centroid of cell (i,j,k)
                    Real yc = ccc(i,j,k,1);
                    Real zc = ccc(i,j,k,2);
 
                    Real delta_x = xf  - xc;
                    Real delta_y = yf  - yc;
                    Real delta_z = 0.5 + zc;
     
                    Real cc_qmax = amrex::max(q(i,j,k,n),q(i,j,k-1,n));
                    Real cc_qmin = amrex::min(q(i,j,k,n),q(i,j,k-1,n));
     
                    // Compute slopes of component "n" of q
                    const auto& slopes_eb_hi = amrex_calc_slopes_extdir_eb(i,j,k,n,q,ccc,
                                               AMREX_D_DECL(fcx,fcy,fcz), flag,
                                               AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                               AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                               AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                               AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));
 
                    //Adding temporal term with the normal derivative to the face
                    Real temp_w = -0.5* vel_c(i,j,k,2) * m_dt * dxinv[2];

                    Real qpls = q(i,j,k  ,n) + (delta_x         ) * slopes_eb_hi[0]
                                             + (delta_y         ) * slopes_eb_hi[1]
                                             - (delta_z + temp_w) * slopes_eb_hi[2];
     
                    qpls = amrex::max(amrex::min(qpls, cc_qmax), cc_qmin);

                    //Adding trans_force
                    if (fq) {
                        qpls += 0.5*m_dt*fq(i,j,k,n);
                    }
 
                    //Adding transverse derivative
                    qpls += - (0.5*m_dt)*(vel_c(i,j,k,0))*dxinv[0]*
                                               (slopes_eb_hi[0]);
                    qpls += - (0.5*m_dt)*(vel_c(i,j,k,1))*dxinv[1]*
                                               (slopes_eb_hi[1]);
     
                    xc = ccc(i,j,k-1,0); // centroid of cell (i,j,k-1)
                    yc = ccc(i,j,k-1,1);
                    zc = ccc(i,j,k-1,2);
     
                    delta_x = xf  - xc;
                    delta_y = yf  - yc;
                    delta_z = 0.5 - zc;

                    // Compute slopes of component "n" of q
                    const auto& slopes_eb_lo = amrex_calc_slopes_extdir_eb(i,j,k-1,n,q,ccc,
                                               AMREX_D_DECL(fcx,fcy,fcz), flag,
                                               AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                               AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                               AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                               AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

                    //Adding temporal term with the normal derivative to the face
                    temp_w = -0.5* vel_c(i,j,k-1,2) * m_dt * dxinv[2];

                    Real qmns = q(i,j,k-1,n) + (delta_x         ) * slopes_eb_lo[0]
                                             + (delta_y         ) * slopes_eb_lo[1]
                                             + (delta_z + temp_w) * slopes_eb_lo[2];
    
                    qmns = amrex::max(amrex::min(qmns, cc_qmax), cc_qmin);

                    if (fq) {
                        qmns += 0.5*m_dt*fq(i,j,k-1,n);
                    }

                    //Adding transverse derivative
                    qmns += - (0.5*m_dt)*(vel_c(i,j,k-1,0))*dxinv[0]*
                                               (slopes_eb_lo[0]);
                    qmns += - (0.5*m_dt)*(vel_c(i,j,k-1,1))*dxinv[1]*
                                               (slopes_eb_lo[1]);

                    if (wmac(i,j,k) > small_vel) {
                        qs = qmns;
                    } else if (wmac(i,j,k) < -small_vel) {
                        qs = qpls;
                    } else {
                        qs = 0.5*(qmns+qpls);
                    }
                }

                fz(i,j,k,n) = wmac(i,j,k) * qs;

           } else {
                fz(i,j,k,n) = 0.0;
           }
        });
#endif

    }
    else // We assume below that the stencil does not need to use hoextrap or extdir boundaries
    {
        // ****************************************************************************
        // Predict to x-faces
        // ****************************************************************************
        amrex::ParallelFor(xbx, ncomp,
        [q,ccc,fcx,flag,umac,small_vel,fx,m_dt,fq,vel_c,dxinv]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
           Real qs;
           if (flag(i,j,k).isConnected(-1,0,0)) 
           {
               Real yf = fcx(i,j,k,0); // local (y,z) of centroid of z-face we are extrapolating to
#if (AMREX_SPACEDIM == 3)
               Real zf = fcx(i,j,k,1);
#endif

               AMREX_D_TERM(Real xc = ccc(i,j,k,0);, // centroid of cell (i,j,k)
                            Real yc = ccc(i,j,k,1);,
                            Real zc = ccc(i,j,k,2););

               AMREX_D_TERM(Real delta_x = 0.5 + xc;,
                            Real delta_y = yf  - yc;,
                            Real delta_z = zf  - zc;);

               Real cc_qmax = amrex::max(q(i,j,k,n),q(i-1,j,k,n));
               Real cc_qmin = amrex::min(q(i,j,k,n),q(i-1,j,k,n));

               // Compute slopes of component "n" of q
               const auto& slopes_eb_hi = amrex_calc_slopes_eb(i,j,k,n,q,ccc,flag);

               Real temp_u = -0.5*vel_c(i,j,k,0)*m_dt*dxinv[0];

#if (AMREX_SPACEDIM == 3)
               Real qpls = q(i  ,j,k,n) - (delta_x + temp_u) * slopes_eb_hi[0]
                                        + (delta_y         ) * slopes_eb_hi[1]
                                        + (delta_z         ) * slopes_eb_hi[2];
#else
               Real qpls = q(i  ,j,k,n) - (delta_x + temp_u) * slopes_eb_hi[0]
                                        + (delta_y         ) * slopes_eb_hi[1];
#endif

               qpls = amrex::max(amrex::min(qpls, cc_qmax), cc_qmin);

               //Adding trans_force
               if (fq) {
                   qpls += 0.5*m_dt*fq(i  ,j,k,n);
               }
   
#if (AMREX_SPACEDIM == 3)
                //Adding transverse derivative
                qpls += - (0.5*m_dt)*(vel_c(i,j,k,1))*dxinv[1]*
                                           (slopes_eb_hi[1]);
                qpls += - (0.5*m_dt)*(vel_c(i,j,k,2))*dxinv[2]*
                                           (slopes_eb_hi[2]);
#else
                //Adding transverse derivative
                qpls += - (0.5*m_dt)*(vel_c(i,j,k,1))*dxinv[1]*
                                           (slopes_eb_hi[1]);
#endif

               AMREX_D_TERM(xc = ccc(i-1,j,k,0);, // centroid of cell (i-1,j,k)
                            yc = ccc(i-1,j,k,1);,
                            zc = ccc(i-1,j,k,2););

               AMREX_D_TERM(delta_x = 0.5 - xc;,
                            delta_y = yf  - yc;,
                            delta_z = zf  - zc;);

               // Compute slopes of component "n" of q
               const auto& slopes_eb_lo = amrex_calc_slopes_eb(i-1,j,k,n,q,ccc,flag);

               //Adding temporal term with the normal derivative to the face
               temp_u = -0.5*vel_c(i-1,j,k,0)*m_dt*dxinv[0];

#if (AMREX_SPACEDIM == 3)
               Real qmns = q(i-1,j,k,n) + (delta_x + temp_u) * slopes_eb_lo[0]
                                        + (delta_y         ) * slopes_eb_lo[1]
                                        + (delta_z         ) * slopes_eb_lo[2];
#else
               Real qmns = q(i-1,j,k,n) + (delta_x + temp_u) * slopes_eb_lo[0]
                                        + (delta_y         ) * slopes_eb_lo[1];
#endif
               qmns = amrex::max(amrex::min(qmns, cc_qmax), cc_qmin);

               //Adding trans_force
               if (fq) {
                   qmns += 0.5*m_dt*fq(i-1,j,k,n);
               }

#if (AMREX_SPACEDIM == 3)
               //Adding transverse derivative
               qmns += - (0.5*m_dt)*(vel_c(i-1,j,k,1)) * dxinv[1] *
                                               (slopes_eb_lo[1]);
               qmns += - (0.5*m_dt)*(vel_c(i-1,j,k,2)) * dxinv[2] *
                                               (slopes_eb_lo[2]);
#else
               //Adding transverse derivative
               qmns += - (0.5*m_dt)*(vel_c(i-1,j,k,1)) * dxinv[1] *
                                               (slopes_eb_lo[1]);
#endif

               if (umac(i,j,k) > small_vel) {
                    qs = qmns;
                } else if (umac(i,j,k) < -small_vel) {
                    qs = qpls;
                } else {
                    qs = 0.5*(qmns+qpls);
                }

                fx(i,j,k,n) = umac(i,j,k) * qs;

           } else {
               fx(i,j,k,n) = 0.0;
           }
        });

        // ****************************************************************************
        // Predict to y-faces
        // ****************************************************************************
        amrex::ParallelFor(ybx, ncomp,
        [q,ccc,fcy,flag,vmac,small_vel,fy,m_dt,fq,vel_c,dxinv]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real qs;
            if (flag(i,j,k).isConnected(0,-1,0)) {

               Real xf = fcy(i,j,k,0); // local (x,z) of centroid of z-face we are extrapolating to
#if (AMREX_SPACEDIM == 3)
               Real zf = fcy(i,j,k,1);
#endif

               AMREX_D_TERM(Real xc = ccc(i,j,k,0);, // centroid of cell (i,j,k)
                            Real yc = ccc(i,j,k,1);,
                            Real zc = ccc(i,j,k,2););

               AMREX_D_TERM(Real delta_x = xf  - xc;,
                            Real delta_y = 0.5 + yc;,
                            Real delta_z = zf  - zc;);

               Real cc_qmax = amrex::max(q(i,j,k,n),q(i,j-1,k,n));
               Real cc_qmin = amrex::min(q(i,j,k,n),q(i,j-1,k,n));

               // Compute slopes of component "n" of q
               const auto& slopes_eb_hi = amrex_calc_slopes_eb(i,j,k,n,q,ccc,flag);

               //Adding temporal term with the normal derivative to the face 
               Real temp_v = -0.5*vel_c(i,j,k,1)*m_dt*dxinv[1];

#if (AMREX_SPACEDIM == 3)
               Real qpls = q(i,j  ,k,n) + (delta_x         ) * slopes_eb_hi[0]
                                        - (delta_y + temp_v) * slopes_eb_hi[1]
                                        + (delta_z         ) * slopes_eb_hi[2];
#else
               Real qpls = q(i,j  ,k,n) + (delta_x         ) * slopes_eb_hi[0]
                                        - (delta_y + temp_v) * slopes_eb_hi[1];
#endif
               qpls = amrex::max(amrex::min(qpls, cc_qmax), cc_qmin);

               //Adding trans_force
               if (fq) {
                   qpls += 0.5*m_dt*fq(i,j  ,k,n);
               }

#if (AMREX_SPACEDIM == 3)
               //Adding transverse derivative
               qpls += - (0.5*m_dt)*(vel_c(i,j,k,0))*dxinv[0]*
                                          (slopes_eb_hi[0]);
               qpls += - (0.5*m_dt)*(vel_c(i,j,k,2))*dxinv[2]*
                                          (slopes_eb_hi[2]);
#else
               //Adding transverse derivative
               qpls += - (0.5*m_dt)*(vel_c(i,j,k,0))*dxinv[0]*
                                          (slopes_eb_hi[0]);
#endif

               AMREX_D_TERM(xc = ccc(i,j-1,k,0);, // centroid of cell (i-1,j,k)
                            yc = ccc(i,j-1,k,1);,
                            zc = ccc(i,j-1,k,2););

               AMREX_D_TERM(delta_x = xf  - xc;,
                            delta_y = 0.5 - yc;,
                            delta_z = zf  - zc;);

               // Compute slopes of component "n" of q
               const auto& slopes_eb_lo = amrex_calc_slopes_eb(i,j-1,k,n,q,ccc,flag);

               //Adding temporal term with the normal derivative to the face 
               temp_v = -0.5*vel_c(i,j-1,k,1)*m_dt*dxinv[1];

#if (AMREX_SPACEDIM == 3)
               Real qmns = q(i,j-1,k,n) + (delta_x         ) * slopes_eb_lo[0]
                                        + (delta_y + temp_v) * slopes_eb_lo[1]
                                        + (delta_z         ) * slopes_eb_lo[2];
#else
               Real qmns = q(i,j-1,k,n) + (delta_x         ) * slopes_eb_lo[0]
                                        + (delta_y + temp_v) * slopes_eb_lo[1];
#endif
               qmns = amrex::max(amrex::min(qmns, cc_qmax), cc_qmin);

               //Adding trans_force
               if (fq) {
                   qmns += 0.5*m_dt*fq(i,j-1,k,n);
               }

#if (AMREX_SPACEDIM == 3)
               //Adding transverse derivative
               qmns += - (0.5*m_dt)*(vel_c(i,j-1,k,0))*dxinv[0]*
                                          (slopes_eb_lo[0]);
               qmns += - (0.5*m_dt)*(vel_c(i,j-1,k,2))*dxinv[2]*
                                          (slopes_eb_lo[2]);
#else
               //Adding transverse derivative
               qmns += - (0.5*m_dt)*(vel_c(i,j-1,k,0))*dxinv[0]*
                                          (slopes_eb_lo[0]);
#endif

               if (vmac(i,j,k) > small_vel) {
                   qs = qmns;
               } else if (vmac(i,j,k) < -small_vel) {
                   qs = qpls;
               } else {
                   qs = 0.5*(qmns+qpls);
               }

               fy(i,j,k,n) = vmac(i,j,k) * qs;

           } else {
               fy(i,j,k,n) = 0.0;
           }
        });

#if (AMREX_SPACEDIM == 3)
        // ****************************************************************************
        // Predict to z-faces
        // ****************************************************************************
        amrex::ParallelFor(zbx, ncomp,
        [q,ccc,fcz,flag,wmac,small_vel,fz,m_dt,fq,vel_c,dxinv]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            if (flag(i,j,k).isConnected(0,0,-1)) {
                Real qs;

                Real xf = fcz(i,j,k,0); // local (x,y) of centroid of z-face we are extrapolating to
                Real yf = fcz(i,j,k,1);
 
                Real xc = ccc(i,j,k,0); // centroid of cell (i,j,k)
                Real yc = ccc(i,j,k,1);
                Real zc = ccc(i,j,k,2);
 
                Real delta_x = xf  - xc;
                Real delta_y = yf  - yc;
                Real delta_z = 0.5 + zc;
     
                Real cc_qmax = amrex::max(q(i,j,k,n),q(i,j,k-1,n));
                Real cc_qmin = amrex::min(q(i,j,k,n),q(i,j,k-1,n));
     
                // Compute slopes of component "n" of q
                const auto& slopes_eb_hi = amrex_calc_slopes_eb(i,j,k,n,q,ccc,flag);
 
                //Adding temporal term with the normal derivative to the face
                Real temp_w = -0.5*vel_c(i,j,k,2)*m_dt*dxinv[2];

                Real qpls = q(i,j,k  ,n) + (delta_x         ) * slopes_eb_hi[0]
                                         + (delta_y         ) * slopes_eb_hi[1]
                                         - (delta_z + temp_w) * slopes_eb_hi[2];
 
                qpls = amrex::max(amrex::min(qpls, cc_qmax), cc_qmin);

                if (fq) {
                    qpls += 0.5*m_dt*fq(i,j,k,n);
                }

                //Adding transverse derivative
                qpls += - (0.5*m_dt)*(vel_c(i,j,k,0))*dxinv[0]*
                                           (slopes_eb_hi[0]);
                qpls += - (0.5*m_dt)*(vel_c(i,j,k,1))*dxinv[1]*
                                           (slopes_eb_hi[1]);
 
                xc = ccc(i,j,k-1,0); // centroid of cell (i,j,k-1)
                yc = ccc(i,j,k-1,1);
                zc = ccc(i,j,k-1,2);
 
                delta_x = xf  - xc;
                delta_y = yf  - yc;
                delta_z = 0.5 - zc;

                // Compute slopes of component "n" of q
                const auto& slopes_eb_lo = amrex_calc_slopes_eb(i,j,k-1,n,q,ccc,flag);

                //Adding temporal term with the normal derivative to the face
                temp_w = -0.5*vel_c(i,j,k-1,2)*m_dt*dxinv[2];

                Real qmns = q(i,j,k-1,n) + (delta_x         ) * slopes_eb_lo[0]
                                         + (delta_y         ) * slopes_eb_lo[1]
                                         + (delta_z + temp_w) * slopes_eb_lo[2];

                qmns = amrex::max(amrex::min(qmns, cc_qmax), cc_qmin);

                if (fq) {
                    qmns += 0.5*m_dt*fq(i,j,k-1,n);
                }

                //Adding transverse derivative
                qmns += - (0.5*m_dt)*(vel_c(i,j,k-1,0))*dxinv[0]*
                                           (slopes_eb_lo[0]);
                qmns += - (0.5*m_dt)*(vel_c(i,j,k-1,1))*dxinv[1]*
                                           (slopes_eb_lo[1]);

                if (wmac(i,j,k) > small_vel) {
                    qs = qmns;
                } else if (wmac(i,j,k) < -small_vel) {
                    qs = qpls;
                } else {
                    qs = 0.5*(qmns+qpls);
                }

                fz(i,j,k,n) = wmac(i,j,k) * qs;

           } else {
                fz(i,j,k,n) = 0.0;
           }
        });
#endif

    } // end of non-extdir section
}
#endif
