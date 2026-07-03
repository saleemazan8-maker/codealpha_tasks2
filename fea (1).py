"""
TASK 2 - Structural FEA of an L-bracket (2D plane stress, linear elastic).
Solver: scikit-fem, constant-strain triangles. Self-contained mesher (SciPy Delaunay).
Includes a cantilever validation against Euler-Bernoulli theory.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from scipy.spatial import Delaunay
from skfem import (MeshTri, Basis, ElementVector, ElementTriP1,
                   BilinearForm, LinearForm, FacetBasis, condense, solve)
from skfem.helpers import sym_grad
import os

os.makedirs("/home/claude/figs", exist_ok=True)

# ============================================================ MATERIAL (6061-T6 Al)
E   = 68900.0     # MPa
NU  = 0.33
SY  = 276.0       # MPa, yield strength
THK = 10.0        # mm, out-of-plane thickness
f   = E / (1.0 - NU**2)

# ============================================================ GEOMETRY (mm)
TARM = 25.0       # leg thickness
H    = 100.0      # vertical leg height
W    = 80.0       # horizontal leg length
RF   = 8.0        # inner-corner fillet radius

def lbracket_polygon(n_arc=16):
    cx, cy = TARM + RF, TARM + RF
    th = np.linspace(-np.pi/2, -np.pi, n_arc)          # arc from (TARM+RF,TARM) to (TARM,TARM+RF)
    arc = np.column_stack([cx + RF*np.cos(th), cy + RF*np.sin(th)])
    pts = [(0, 0), (W, 0), (W, TARM), (TARM+RF, TARM)]
    pts += list(map(tuple, arc))
    pts += [(TARM, H), (0, H)]
    return np.array(pts)

# ============================================================ MESHER (Delaunay + clip)
def build_mesh(poly, h=1.8):
    path = Path(poly)
    # boundary nodes sampled along each edge at spacing ~h
    bpts = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i+1) % len(poly)]
        L = np.hypot(*(b-a))
        n = max(int(np.ceil(L/h)), 1)
        for k in range(n):
            bpts.append(a + (b-a)*k/n)
    bpts = np.array(bpts)
    # interior grid nodes inside the polygon
    xs = np.arange(poly[:,0].min()+h/2, poly[:,0].max(), h)
    ys = np.arange(poly[:,1].min()+h/2, poly[:,1].max(), h)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    inside = path.contains_points(grid, radius=-h*0.35)
    pts = np.vstack([bpts, grid[inside]])
    # remove duplicate coincident nodes (boundary corners etc.)
    key = np.round(pts / 1e-6).astype(np.int64)
    _, uidx = np.unique(key, axis=0, return_index=True)
    pts = pts[np.sort(uidx)]
    # triangulate, keep triangles with centroid inside the domain
    tri = Delaunay(pts)
    cents = pts[tri.simplices].mean(axis=1)
    keep = path.contains_points(cents, radius=-1e-6)
    cells = tri.simplices[keep]
    # prune orphan nodes (not referenced by any kept triangle) and reindex
    used = np.unique(cells)
    remap = -np.ones(len(pts), dtype=np.int64); remap[used] = np.arange(len(used))
    pts = pts[used]
    cells = remap[cells]
    # orient CCW (positive area)
    p = pts[cells]
    area2 = ((p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])
             - (p[:,2,0]-p[:,0,0])*(p[:,1,1]-p[:,0,1]))
    flip = area2 < 0
    cells[flip] = cells[flip][:, ::-1]
    m = MeshTri(pts.T, cells.T.astype(np.int64))
    m = m.with_boundaries({
        "top":  lambda x: np.isclose(x[1], H),     # vertical-leg top edge (bolted)
        "back": lambda x: np.isclose(x[0], 0.0),   # vertical-leg back face
        "end":  lambda x: np.isclose(x[0], W),     # horizontal-leg end face (loaded)
    })
    return m

# ============================================================ FE FORMS (plane stress)
@BilinearForm
def stiffness(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    exx, eyy, exy = eu[0,0], eu[1,1], eu[0,1]
    sxx = f*(exx + NU*eyy)
    syy = f*(NU*exx + eyy)
    sxy = f*(1.0-NU)*exy
    return sxx*ev[0,0] + syy*ev[1,1] + 2.0*sxy*ev[0,1]

def solve_case(mesh, fixed, load_edge, Fx, Fy):
    e = ElementVector(ElementTriP1())
    basis = Basis(mesh, e)
    K = stiffness.assemble(basis)
    fb = FacetBasis(mesh, e, facets=mesh.boundaries[load_edge])
    L = float(np.sum(_facet_lengths(mesh, load_edge)))
    px, py = Fx/(THK*L), Fy/(THK*L)
    @LinearForm
    def traction(v, w):
        return px*v[0] + py*v[1]
    b = traction.assemble(fb)
    D = basis.get_dofs(fixed)
    u = solve(*condense(K, b, D=D))
    return basis, u, L

def _facet_lengths(mesh, name):
    fac = mesh.facets[:, mesh.boundaries[name]]
    p0 = mesh.p[:, fac[0]]; p1 = mesh.p[:, fac[1]]
    return np.hypot(p1[0]-p0[0], p1[1]-p0[1])

# ============================================================ POST-PROCESS
def fields(basis, u):
    m = basis.mesh
    wu = basis.interpolate(u)
    g = wu.grad                      # (2,2,nelem,nqp)
    exx = g[0,0,:,0]; eyy = g[1,1,:,0]
    exy = 0.5*(g[0,1,:,0] + g[1,0,:,0])
    sxx = f*(exx + NU*eyy)
    syy = f*(NU*exx + eyy)
    sxy = f*(1.0-NU)*exy
    svm = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2)   # per element (MPa)
    ux = u[basis.nodal_dofs[0]]
    uy = u[basis.nodal_dofs[1]]
    dmag = np.sqrt(ux**2 + uy**2)    # per node (mm)
    return svm, dmag, ux, uy

# ============================================================ VALIDATION (cantilever)
def validate():
    Lb, hb, tb = 100.0, 10.0, 10.0
    poly = np.array([(0,0),(Lb,0),(Lb,hb),(0,hb)])
    # simple structured-ish mesh via the same builder
    m = build_mesh.__wrapped__ if hasattr(build_mesh,"__wrapped__") else None
    # build a rectangular mesh directly
    nx, ny = 80, 8
    xs = np.linspace(0,Lb,nx+1); ys = np.linspace(0,hb,ny+1)
    gx,gy = np.meshgrid(xs,ys); pts=np.column_stack([gx.ravel(),gy.ravel()])
    tri = Delaunay(pts); cells=tri.simplices
    p=pts[cells]
    a2=((p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])-(p[:,2,0]-p[:,0,0])*(p[:,1,1]-p[:,0,1]))
    cells[a2<0]=cells[a2<0][:, ::-1]
    mb = MeshTri(pts.T, cells.T.astype(np.int64)).with_boundaries({
        "fix": lambda x: np.isclose(x[0],0.0),
        "tip": lambda x: np.isclose(x[0],Lb)})
    e=ElementVector(ElementTriP1()); basis=Basis(mb,e)
    K=stiffness.assemble(basis)
    fb=FacetBasis(mb,e,facets=mb.boundaries["tip"])
    Fz=-100.0; L=hb
    py=Fz/(tb*L)
    @LinearForm
    def trac(v,w): return 0.0*v[0]+py*v[1]
    b=trac.assemble(fb)
    u=solve(*condense(K,b,D=basis.get_dofs("fix")))
    uy=u[basis.nodal_dofs[1]]
    fem_def=np.abs(uy).max()
    I=tb*hb**3/12.0
    eb_def=abs(Fz)*Lb**3/(3*E*I)
    svm,_,_,_=fields(basis,u)
    sigma_bending=6*abs(Fz)*Lb/(tb*hb**2)
    return fem_def, eb_def, svm.max(), sigma_bending

fem_def, eb_def, fem_sig, eb_sig = validate()
print("VALIDATION (cantilever 100x10x10, tip load 100 N):")
print(f"  tip deflection : FEM {fem_def:.4f} mm | Euler-Bernoulli {eb_def:.4f} mm "
      f"| diff {100*(fem_def-eb_def)/eb_def:+.1f}%")
print(f"  peak stress    : FEM {fem_sig:.1f} MPa | beam theory {eb_sig:.1f} MPa\n")

# ============================================================ RUN THE 3 CASES
mesh = build_mesh(lbracket_polygon(), h=1.7)
print(f"Mesh: {mesh.p.shape[1]} nodes, {mesh.t.shape[1]} triangular elements\n")

CASES = [
    dict(name="Case 1", desc="Vertical tip load, bolted flange fixed",
         fixed="top",  load="end", Fx=0.0,   Fy=-600.0),
    dict(name="Case 2", desc="Horizontal pull-out load, bolted flange fixed",
         fixed="top",  load="end", Fx=600.0, Fy=0.0),
    dict(name="Case 3", desc="Vertical tip load, full back face fixed",
         fixed="back", load="end", Fx=0.0,   Fy=-600.0),
]

results = []
for c in CASES:
    basis, u, L = solve_case(mesh, c["fixed"], c["load"], c["Fx"], c["Fy"])
    svm, dmag, ux, uy = fields(basis, u)
    smax = float(svm.max()); dmax = float(dmag.max())
    fos = SY/smax
    results.append(dict(c=c, basis=basis, u=u, svm=svm, dmag=dmag,
                        ux=ux, uy=uy, smax=smax, dmax=dmax, fos=fos, L=L))
    print(f"{c['name']}: {c['desc']}")
    print(f"   loaded-edge length {L:.1f} mm  |  max von Mises {smax:6.1f} MPa  "
          f"|  max deflection {dmax:.4f} mm  |  min FoS {fos:.2f}\n")

np.save("/home/claude/figs/_results.npy", np.array(
    [[r['smax'], r['dmax'], r['fos']] for r in results]))

# ============================================================ FIGURES
def tri_obj(m):
    return mtri.Triangulation(m.p[0], m.p[1], m.t.T)

def save_setup_figure():
    poly = lbracket_polygon()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    arrows = [((W, TARM/2), (0, -1), "600 N"),
              ((W, TARM/2), (1, 0), "600 N"),
              ((W, TARM/2), (0, -1), "600 N")]
    fixed_edges = ["top", "back", "back"]  # case3 fixes back; case1,2 fix top
    fixed_edges = ["top", "top", "back"]
    for ax, c, arr, fe in zip(axes, CASES, arrows, fixed_edges):
        ax.fill(poly[:,0], poly[:,1], color="#c9d6e5", ec="#3b5b7a", lw=1.5)
        # fixed edge hatch
        if fe == "top":
            ax.plot([0, TARM], [H, H], color="#b00", lw=4)
            for xx in np.linspace(0, TARM, 6):
                ax.plot([xx, xx-3], [H, H+5], color="#b00", lw=1)
        else:
            ax.plot([0, 0], [0, H], color="#b00", lw=4)
            for yy in np.linspace(0, H, 11):
                ax.plot([0, -4], [yy, yy+3], color="#b00", lw=1)
        (ox, oy), (dx, dy), lbl = arr
        ax.annotate("", xy=(ox+dx*22, oy+dy*22), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2.5))
        ax.text(ox+dx*24+ (3 if dx else 0), oy+dy*24, lbl, color="#c0392b",
                fontsize=9, ha="left", va="center")
        ax.set_title(f"{c['name']}\n{c['desc']}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xlim(-12, W+18); ax.set_ylim(-12, H+14)
        ax.set_xlabel("mm"); ax.grid(alpha=0.2)
    fig.suptitle("L-bracket — geometry, loads & boundary conditions  "
                 "(6061-T6 Al, t = 10 mm, inner fillet R8)", fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig("/home/claude/figs/setup.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

def save_mesh_figure():
    fig, ax = plt.subplots(figsize=(5.2, 6))
    ax.triplot(tri_obj(mesh), color="#5b6b7b", lw=0.4)
    ax.set_aspect("equal"); ax.set_title(
        f"Finite-element mesh\n{mesh.p.shape[1]} nodes, {mesh.t.shape[1]} CST elements",
        fontsize=10)
    ax.set_xlabel("mm"); ax.set_ylabel("mm")
    fig.tight_layout()
    fig.savefig("/home/claude/figs/mesh.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

def save_case_figure(idx, r):
    m = mesh
    scale = 12.0 / max(r["dmax"], 1e-9)
    dp = np.vstack([m.p[0] + scale*r["ux"], m.p[1] + scale*r["uy"]])
    tdef = mtri.Triangulation(dp[0], dp[1], m.t.T)
    tund = tri_obj(m)
    fos_field = np.clip(SY/np.maximum(r["svm"], 1e-6), 0, 8)

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 5.2))

    p0 = ax[0].tripcolor(tdef, facecolors=r["svm"], cmap="inferno", shading="flat")
    ax[0].triplot(tund, color="0.7", lw=0.3)
    ax[0].set_title(f"von Mises stress (deformed ×{scale:.0f})\nmax = {r['smax']:.1f} MPa")
    fig.colorbar(p0, ax=ax[0], fraction=0.046, pad=0.02, label="MPa")

    p1 = ax[1].tripcolor(tund, r["dmag"], cmap="viridis", shading="gouraud")
    ax[1].set_title(f"Displacement magnitude\nmax = {r['dmax']:.3f} mm")
    fig.colorbar(p1, ax=ax[1], fraction=0.046, pad=0.02, label="mm")

    p2 = ax[2].tripcolor(tund, facecolors=fos_field, cmap="RdYlGn", shading="flat",
                         vmin=1, vmax=8)
    ax[2].set_title(f"Factor of safety (Sy/σvm)\nmin = {r['fos']:.2f}")
    fig.colorbar(p2, ax=ax[2], fraction=0.046, pad=0.02, label="FoS")

    for a in ax:
        a.set_aspect("equal"); a.set_xlabel("mm")
    fig.suptitle(f"{r['c']['name']} — {r['c']['desc']}", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"/home/claude/figs/case{idx+1}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

save_setup_figure()
save_mesh_figure()
for i, r in enumerate(results):
    save_case_figure(i, r)
print("Figures written to /home/claude/figs/")
