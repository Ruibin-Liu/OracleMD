// E0g kernels v2: arithmetic ladder + compact list (W-block verified 3.1e-14)
#define RSQRTPI 0.5641895835477563

__device__ __forceinline__ double W0(double t) {
    return ((((((((((((((((((+2.19935009363420147e-10*t-1.19435620325330841e-09)*t+4.42788930256360589e-09)*t-1.86014220734415395e-08)*t+8.07600742878595086e-08)*t-3.35144993111066006e-07)*t+1.34243296404639601e-06)*t-5.20962084392019835e-06)*t+1.95266080384681256e-05)*t-7.04679827391670090e-05)*t+2.44039031762325503e-04)*t-8.07782029369062682e-04)*t+2.54317035818231582e-03)*t-7.56936980337291881e-03)*t+2.11329450976001504e-02)*t-5.47745886538915511e-02)*t+1.29913948997565865e-01)*t-2.75979518741850893e-01)*t+5.06937650293145192e-01);
}
__device__ __forceinline__ double W1(double t) {
    return ((((((((((((((+1.82218060187137474e-10*t-1.00905589840545899e-09)*t+4.71420328519759667e-09)*t-2.43522908366651598e-08)*t+1.24068122897698662e-07)*t-6.12088052588045740e-07)*t+2.93863938978228677e-06)*t-1.37115593164665698e-05)*t+6.20318158827869387e-05)*t-2.71412140420430772e-04)*t+1.14507274914490368e-03)*t-4.64149438073509744e-03)*t+1.79958529184971079e-02)*t-6.63648771065850906e-02)*t+2.31087258730392209e-01);
}
__device__ __forceinline__ double W2(double t) {
    return ((((((((((((((((((+1.16099213085226341e-10*t-4.42548124312728053e-10)*t+1.04009121870080483e-09)*t-3.52776855919964062e-09)*t+1.32084281919475480e-08)*t-4.53816976253245234e-08)*t+1.52452698965931662e-07)*t-5.09310289079644697e-07)*t+1.68164518958108709e-06)*t-5.47956945116423911e-06)*t+1.76184602562209627e-05)*t-5.58730192110227247e-05)*t+1.74667247329627305e-04)*t-5.37951714317812420e-04)*t+1.63125725782422670e-03)*t-4.86684252617713281e-03)*t+1.42753120049775029e-02)*t-4.11310350467865293e-02)*t+1.16302707210247463e-01);
}

__device__ __forceinline__ double erfc_poly(double x, double emx2) {
    double w;
    if (x < 1.5)      { double t = x*1.3333333333333333 - 1.0;              w = W0(t); }
    else if (x < 3.0) { double t = (x-1.5)*1.3333333333333333 - 1.0;        w = W1(t); }
    else              { double xc = x > 6.5 ? 6.5 : x;
                        double t = (xc-3.0)*0.5714285714285714 - 1.0;      w = W2(t); }
    return emx2 * w;
}

extern "C" __global__ void k0_base(
    const double* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    const double* __restrict__ q, const double* __restrict__ sig, const double* __restrict__ eps,
    const double* __restrict__ se, double* __restrict__ F,
    int N, int R, int maxnb, double rc2, double alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    double xi = x[(a*R+r)*3], yi = x[(a*R+r)*3+1], zi = x[(a*R+r)*3+2];
    double qi = q[a], si = sig[a], ei = eps[a];
    double fx=0., fy=0., fz=0.;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a*maxnb + k];
        double dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < rc2) {
            double invr2 = 1.0/r2, invr = 1.0/sqrt(r2);
            double s = 0.5*(si+sig[j]), e = sqrt(ei*eps[j]);
            double sr2 = s*s*invr2, sr6 = sr2*sr2*sr2;
            double flj = 24.0*e*(2.0*sr6*sr6 - sr6)*invr2;
            double emx2 = exp(-alpha*alpha*r2);
            double ec = erfc_poly(alpha/invr, emx2);
            double fc = qi*q[j]*(ec*invr2 + 2.0*alpha*RSQRTPI*emx2*invr);
            double f = flj + fc;
            fx += f*dx; fy += f*dy; fz += f*dz;
        }
    }
    F[(a*R+r)*3] = fx; F[(a*R+r)*3+1] = fy; F[(a*R+r)*3+2] = fz;
}

extern "C" __global__ void k1_arith(
    const double* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    const double* __restrict__ q, const double* __restrict__ sig, const double* __restrict__ eps,
    const double* __restrict__ se, double* __restrict__ F,
    int N, int R, int maxnb, double rc2, double alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    double xi = x[(a*R+r)*3], yi = x[(a*R+r)*3+1], zi = x[(a*R+r)*3+2];
    double qi = q[a], si = sig[a], ei = eps[a];
    double fx=0., fy=0., fz=0.;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a*maxnb + k];
        double dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < rc2) {
            double invr2 = 1.0/r2;
            double rr = sqrt(r2);
            double invr = rr*invr2;
            double s = 0.5*(si+sig[j]), e = sqrt(ei*eps[j]);
            double sr2 = s*s*invr2, sr6 = sr2*sr2*sr2;
            double flj = 24.0*e*(2.0*sr6*sr6 - sr6)*invr2;
            double emx2 = exp(-alpha*alpha*r2);
            double ec = erfc_poly(alpha*rr, emx2);
            double fc = qi*q[j]*(ec*invr2 + 2.0*alpha*RSQRTPI*emx2*invr);
            double f = flj + fc;
            fx += f*dx; fy += f*dy; fz += f*dz;
        }
    }
    F[(a*R+r)*3] = fx; F[(a*R+r)*3+1] = fy; F[(a*R+r)*3+2] = fz;
}

// K2: sqrt(eps) precompute + LJ/coul separate accumulators + end fold
// (inner e uses se[j] only; sei applied once at the end)
extern "C" __global__ void k2_prefold(
    const double* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    const double* __restrict__ q, const double* __restrict__ sig, const double* __restrict__ eps,
    const double* __restrict__ se, double* __restrict__ F,
    int N, int R, int maxnb, double rc2, double alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    double xi = x[(a*R+r)*3], yi = x[(a*R+r)*3+1], zi = x[(a*R+r)*3+2];
    double qi = q[a], si = sig[a], sei = se[a];
    double lx=0., ly=0., lz=0., cx=0., cy=0., cz=0.;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a*maxnb + k];
        double dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < rc2) {
            double invr2 = 1.0/r2;
            double rr = sqrt(r2);
            double invr = rr*invr2;
            double s = 0.5*(si+sig[j]), e = se[j];
            double sr2 = s*s*invr2, sr6 = sr2*sr2*sr2;
            double flj = 24.0*e*(2.0*sr6*sr6 - sr6)*invr2;
            double emx2 = exp(-alpha*alpha*r2);
            double ec = erfc_poly(alpha*rr, emx2);
            double fc = q[j]*(ec*invr2 + 2.0*alpha*RSQRTPI*emx2*invr);
            lx += flj*dx; ly += flj*dy; lz += flj*dz;
            cx += fc*dx; cy += fc*dy; cz += fc*dz;
        }
    }
    F[(a*R+r)*3]   = sei*lx + qi*cx;
    F[(a*R+r)*3+1] = sei*ly + qi*cy;
    F[(a*R+r)*3+2] = sei*lz + qi*cz;
}

// K3: K2 + unroll x4 with independent REGISTER accumulators (if-chains)
extern "C" __global__ void k3_unroll(
    const double* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    const double* __restrict__ q, const double* __restrict__ sig, const double* __restrict__ eps,
    const double* __restrict__ se, double* __restrict__ F,
    int N, int R, int maxnb, double rc2, double alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    double xi = x[(a*R+r)*3], yi = x[(a*R+r)*3+1], zi = x[(a*R+r)*3+2];
    double qi = q[a], si = sig[a], sei = se[a];
    double lx0=0.,ly0=0.,lz0=0.,cx0=0.,cy0=0.,cz0=0.;
    double lx1=0.,ly1=0.,lz1=0.,cx1=0.,cy1=0.,cz1=0.;
    double lx2=0.,ly2=0.,lz2=0.,cx2=0.,cy2=0.,cz2=0.;
    double lx3=0.,ly3=0.,lz3=0.,cx3=0.,cy3=0.,cz3=0.;
    int nb = ncount[a];
    int k = 0;
    for (; k + 4 <= nb; k += 4) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            int j = nlist[a*maxnb + k + u];
            double dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < rc2) {
                double invr2 = 1.0/r2;
                double rr = sqrt(r2);
                double invr = rr*invr2;
                double s = 0.5*(si+sig[j]), e = se[j];
                double sr2 = s*s*invr2, sr6 = sr2*sr2*sr2;
                double flj = 24.0*e*(2.0*sr6*sr6 - sr6)*invr2;
                double emx2 = exp(-alpha*alpha*r2);
                double ec = erfc_poly(alpha*rr, emx2);
                double fc = q[j]*(ec*invr2 + 2.0*alpha*RSQRTPI*emx2*invr);
                if (u == 0) { lx0 += flj*dx; ly0 += flj*dy; lz0 += flj*dz; cx0 += fc*dx; cy0 += fc*dy; cz0 += fc*dz; }
                else if (u == 1) { lx1 += flj*dx; ly1 += flj*dy; lz1 += flj*dz; cx1 += fc*dx; cy1 += fc*dy; cz1 += fc*dz; }
                else if (u == 2) { lx2 += flj*dx; ly2 += flj*dy; lz2 += flj*dz; cx2 += fc*dx; cy2 += fc*dy; cz2 += fc*dz; }
                else { lx3 += flj*dx; ly3 += flj*dy; lz3 += flj*dz; cx3 += fc*dx; cy3 += fc*dy; cz3 += fc*dz; }
            }
        }
    }
    for (; k < nb; ++k) {
        int j = nlist[a*maxnb + k];
        double dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < rc2) {
            double invr2 = 1.0/r2;
            double rr = sqrt(r2);
            double invr = rr*invr2;
            double s = 0.5*(si+sig[j]), e = se[j];
            double sr2 = s*s*invr2, sr6 = sr2*sr2*sr2;
            double flj = 24.0*e*(2.0*sr6*sr6 - sr6)*invr2;
            double emx2 = exp(-alpha*alpha*r2);
            double ec = erfc_poly(alpha*rr, emx2);
            double fc = q[j]*(ec*invr2 + 2.0*alpha*RSQRTPI*emx2*invr);
            lx0 += flj*dx; ly0 += flj*dy; lz0 += flj*dz;
            cx0 += fc*dx; cy0 += fc*dy; cz0 += fc*dz;
        }
    }
    F[(a*R+r)*3]   = sei*(lx0+lx1+lx2+lx3) + qi*(cx0+cx1+cx2+cx3);
    F[(a*R+r)*3+1] = sei*(ly0+ly1+ly2+ly3) + qi*(cy0+cy1+cy2+cy3);
    F[(a*R+r)*3+2] = sei*(lz0+lz1+lz2+lz3) + qi*(cz0+cz1+cz2+cz3);
}

// K4: compact in-rc list, no mask branch (production per-step shape;
// ring pairs never enter the force loop)
extern "C" __global__ void k4_compact(
    const double* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    const double* __restrict__ q, const double* __restrict__ sig, const double* __restrict__ eps,
    const double* __restrict__ se, double* __restrict__ F,
    int N, int R, int maxnb, double rc2, double alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    double xi = x[(a*R+r)*3], yi = x[(a*R+r)*3+1], zi = x[(a*R+r)*3+2];
    double qi = q[a], si = sig[a], sei = se[a];
    double lx=0., ly=0., lz=0., cx=0., cy=0., cz=0.;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a*maxnb + k];
        double dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
        double r2 = dx*dx + dy*dy + dz*dz;
        double invr2 = 1.0/r2;
        double rr = sqrt(r2);
        double invr = rr*invr2;
        double s = 0.5*(si+sig[j]), e = se[j];
        double sr2 = s*s*invr2, sr6 = sr2*sr2*sr2;
        double flj = 24.0*e*(2.0*sr6*sr6 - sr6)*invr2;
        double emx2 = exp(-alpha*alpha*r2);
        double ec = erfc_poly(alpha*rr, emx2);
        double fc = q[j]*(ec*invr2 + 2.0*alpha*RSQRTPI*emx2*invr);
        lx += flj*dx; ly += flj*dy; lz += flj*dz;
        cx += fc*dx; cy += fc*dy; cz += fc*dz;
    }
    F[(a*R+r)*3]   = sei*lx + qi*cx;
    F[(a*R+r)*3+1] = sei*ly + qi*cy;
    F[(a*R+r)*3+2] = sei*lz + qi*cz;
}

// K5: compact in-rc list + multiply-by-zero mask (spec 5.2 pillar-5 compliant;
// drifted-past-rc pairs contribute exactly +0, list content stays out of semantics)
extern "C" __global__ void k5_compact_masked(
    const double* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    const double* __restrict__ q, const double* __restrict__ sig, const double* __restrict__ eps,
    const double* __restrict__ se, double* __restrict__ F,
    int N, int R, int maxnb, double rc2, double alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    double xi = x[(a*R+r)*3], yi = x[(a*R+r)*3+1], zi = x[(a*R+r)*3+2];
    double qi = q[a], si = sig[a], sei = se[a];
    double lx=0., ly=0., lz=0., cx=0., cy=0., cz=0.;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a*maxnb + k];
        double dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
        double r2 = dx*dx + dy*dy + dz*dz;
        double inside = r2 < rc2 ? 1.0 : 0.0;   // exact 0/1, multiply-mask (pillar 5)
        double invr2 = 1.0/r2;
        double rr = sqrt(r2);
        double invr = rr*invr2;
        double s = 0.5*(si+sig[j]), e = se[j];
        double sr2 = s*s*invr2, sr6 = sr2*sr2*sr2;
        double flj = 24.0*e*(2.0*sr6*sr6 - sr6)*invr2*inside;
        double emx2 = exp(-alpha*alpha*r2);
        double ec = erfc_poly(alpha*rr, emx2);
        double fc = q[j]*(ec*invr2 + 2.0*alpha*RSQRTPI*emx2*invr)*inside;
        lx += flj*dx; ly += flj*dy; lz += flj*dz;
        cx += fc*dx; cy += fc*dy; cz += fc*dz;
    }
    F[(a*R+r)*3]   = sei*lx + qi*cx;
    F[(a*R+r)*3+1] = sei*ly + qi*cy;
    F[(a*R+r)*3+2] = sei*lz + qi*cz;
}
