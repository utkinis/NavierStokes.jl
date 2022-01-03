# NavierStokes.jl

NavierStokes.jl is a technical demo featuring Navier-Stokes solver based on a projection method in 200 lines of code. The code works both on multi-core CPU and GPU architectures relying on the [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) Julia package. 

The animation depicts von Karman vortex sheet for Reynolds number of 1e6 resolved on a numerical grid composed of 2559x4095 grid points (running the [`NavierStokes_Re6.jl`](scripts/NavierStokes_Re6.jl) on a Nvidia Tesla V100 GPU).

<img src="./vis/anim/ns_re1e6_2559x4095_small.gif" alt="von Karman vortex sheet" width="800">
