using Plots
using ParallelStencil


"""
    bilinearly interpolates the value of A at point (x,y)

    A:   quantity to interpolate  (nx*ny grid)
    Xc:  x coordinate grid values (nx  vector)
    Yc:  y coordinate grid values (ny  vector)
    il:  grid index for x coordinate left of (x,y). Xc[il] <= x < Xc[il+1]
    jl:  grid index for y coordinate under (x,y).   Yc[il] <= y < Yc[jl+1]
"""
function interpolate_bilinear(A, Xc, Yc, x, y, il, jl)

    # interpolate in x direction
    x1, x2 = Xc[il], Xc[il+1]
    sigma1 = (x2-x)/(x2-x1) * A[il, jl  ] + (x-x1)/(x2-x1) * A[il+1, jl  ]
    sigma2 = (x2-x)/(x2-x1) * A[il, jl+1] + (x-x1)/(x2-x1) * A[il+1, jl+1]

    # interpolate in y direction
    y1, y2 = Yc[jl], Yc[jl+1]
    sigma  = (y2-y)/(y2-y1) * sigma1 + (y-y1)/(y2-y1) * sigma2

    return sigma
end


"""
    advects quantity A using Stam's method of characteristics

    Vx, Vy:  velocity field
    dt:      timestep
    dx, dy:  spatial discretization cell sizes
    nx, ny:  grid size
    Lx, Ly:  grid length
    xc, yc:  grid points
"""
function stams_method(A, Vx, Vy, dt, dx, dy, nx, ny, Lx, Ly, xc, yc)

    # initialize temporary grid
    advected=zeros(nx,ny)

    # ignore boundaries
    for i=2:nx-1
        for j=2:ny-1

            # get velocities
            x_vel = Vx[i,j] # 0.5*(Vx[i,j] + Vx[1+i,j])
            y_vel = Vy[i,j] # 0.5*(Vy[i,j] + Vy[i,1+j])

            # get origin of current point
            x_origin = xc[i] - dt * x_vel
            y_origin = yc[j] - dt * y_vel

            # skip if origin points are off the grid
            if (isnan(x_origin) || isnan(y_origin) || x_origin < 0 || x_origin > Lx || y_origin < 0 || y_origin > Ly)
                continue
            end

            # find grid points surrounding (x_origin, y_origin)
            i_left  = 1+Int(floor((x_origin) / dx))
            i_right = i_left + 1
            j_low   = 1+Int(floor((y_origin) / dy))
            j_top   = j_low + 1 

            # skip if origin points are off the grid
            if (i_left < 1 || i_right > nx || j_low < 1 || j_top > ny )
                continue
            end

            # bilinear interpolation
            origin_A = interpolate_bilinear(A, xc, yc, x_origin, y_origin, i_left, j_low) 

            # update advection grid
            advected[i, j] = origin_A

        end
    end

    return advected
end

"""
    calculates the divergence

    Vx, Vy:  velocity field
    nx, ny:  grid size
    dx, dy:  spatial discretization cell size
"""
function divergence(Vx, Vy, nx, ny, dx, dy)

    # initialization
    div = zeros(nx, ny)
    a   = 0.5

    # ignore boundaries
    for i=2:nx-1
        for j=2:ny-1
            div[i, j] = a * ((Vx[i+1, j] - Vx[i, j]) / dx + (Vy[i, j+1] - Vy[i, j]) / dy)
        end 
    end

    return div
end

"""
    calculates the pressure array using the Poisson equation (goal: divergence-freedom)

    divergence:  divergence
    nx, ny:      grid size
    dx, dy:      spatial discretization cell sizes
    max_iter:    maximal number of iterations
    TODO:        add tolerance and convergence condition
"""
function pressure(divergence, nx, ny, dx, dy, max_iter)
    
    # initialization
    pressure = ones(nx, ny)
    tmp = zeros(nx, ny)
    alpha = 2 / (dx^2) + 2 / (dy^2)
    
    # iterative solver according to Glimberg paper
    for it=1:max_iter # TODO: add convergence condition

        # ignore boundaries
        for i=2:nx-1       
            for j=2:ny-1  

                # update viscous velocity
                pressure[i,j] = ((tmp[i-1, j] + tmp[i+1, j]) / (dx^2) + (tmp[i, j-1] + tmp[i, j+1]) / (dy^2) - divergence[i, j]) / alpha

            end
        end

        # pointer swap
        tmp, pressure = pressure, tmp 
    end

    return pressure
end

"""
    implements no flux boundary conditions

    A:   grid
"""
function bc_noflux!(A; b=1)
    A[  1, 2:end-1] .= b * A[    2, 2:end-1]
    A[end, 2:end-1] .= b * A[end-1, 2:end-1]
    A[2:end-1,   1] .= b * A[2:end-1,     2]
    A[2:end-1, end] .= b * A[2:end-1, end-1]
end

"""
    implement constant boundaries
    
    A:   grid
    c:   constant
"""
function bc_constant!(A, c)
    A[  1, 2:end-1] .= c
    A[end, 2:end-1] .= c
    A[2:end-1,   1] .= c 
    A[2:end-1, end] .= c 
end


@views function smoke_2D()

    # Parameters. TODO
    external_force_parameter =  0.0
    smoke_density_parameter  =  0.05
    buoyancy_parameter       =  1
    viscosity                =  1
    T0                       =  0.2  # TODO: what value

    # Physics
    Lx, Ly   = 10.0, 10.0
    D        = 1.0
    ttot     = 0.5

    # Numerics
    nx, ny   = 64, 64
    nout     = 2
    num_iter = 30

    # Derived numerics
    dx, dy   = Lx/nx, Ly/ny
    dt       = min(dx, dy)^2/D/4.1
    nt       = cld(ttot, dt)
    xc, yc   = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)

    # Array allocation 
    C  = zeros(nx, ny)  # concentration/density
    T  = zeros(nx, ny)  # temperature
    Vx = zeros(nx, ny)  # x velocity
    Vy = zeros(nx, ny)  # y velocity
    P  = zeros(nx, ny)  # pressure
    # F  = zeros(nx, ny)  # external force

    # TODO: Initialization?
    # C = exp.(.-(xc .- Lx/2).^2 .-(yc' .- Ly/3).^2)    
    T = 0.5*exp.(.- ((xc .- Lx/2) .* 4) .^ 2 .- ((yc' .- Ly/2) .* 4) .^ 2)
    T[:,1  ] .= 1
    T[:,end] .= -1


    del  = 4 # parameter
    #C[nx÷2-del:nx÷2+del, 1:4] .= 1.0

    # Initialize animation
    anim = Animation()

    # Time loop
    for it = 1:nt

        # Acceleration grids
        dVx_dt  = zeros(nx-2, ny-2)
        dVy_dt  = zeros(nx-2, ny-2)

        # Add diffusion term
        dVx_dt .+= viscosity .* (diff(diff(Vx[:, 2:end-1], dims=1), dims=1) / (dx^2) 
                               + diff(diff(Vx[2:end-1, :], dims=2), dims=2) / (dy^2))
        dVy_dt .+= viscosity .* (diff(diff(Vy[:, 2:end-1], dims=1), dims=1) / (dx^2) 
                               + diff(diff(Vy[2:end-1, :], dims=2), dims=2) / (dy^2))

        # Add advection term  (we average over the velocity fields left-right)
        dVx_dt .-=  ((Vx[2:end-1, 2:end-1] .* (diff(Vx, dims=1)[1:end-1, 2:end-1] + diff(Vx, dims=1)[2:end, 2:end-1]) / (2*dx))
                   + (Vy[2:end-1, 2:end-1] .* (diff(Vx, dims=2)[2:end-1, 1:end-1] + diff(Vx, dims=2)[2:end-1, 2:end]) / (2*dy)))
        dVy_dt .-=  ((Vx[2:end-1, 2:end-1] .* (diff(Vy, dims=1)[1:end-1, 2:end-1] + diff(Vy, dims=1)[2:end, 2:end-1]) / (2*dx))
                   + (Vy[2:end-1, 2:end-1] .* (diff(Vy, dims=2)[2:end-1, 1:end-1] + diff(Vy, dims=2)[2:end-1, 2:end]) / (2*dy)))
 
        # Add external (vertical) forces (Boussinesq approximation)
        dVx_dt .+= 0
        dVy_dt .+= smoke_density_parameter .* C[2:end-1, 2:end-1] .+ buoyancy_parameter .* (T[2:end-1, 2:end-1] .- T0)
        
        # Update velocities before pressure update
        Vx[2:end-1, 2:end-1]     .+= dt * dVx_dt
        Vy[2:end-1, 2:end-1]     .+= dt * dVy_dt

        # Compute divergence
        div = divergence(Vx, Vy, nx, ny, dx, dy)   

        # Compute pressure using Poisson equations
        P  = pressure(div, nx, ny, dx, dy, 4000)  # TODO ?

        # Boundaries. TODO
        #bc_constant!(P, 1)
        bc_noflux!(P)

        # Subtract gradP from vV   (average for gradient)
        Vx[2:end-1, 2:end-1]  .-= 0.5/dx .* (P[3:end, 2:end-1] .- P[1:end-2, 2:end-1]) # (diff(P, dims=1)[1:end-1, 2:end-1] + diff(P, dims=1)[2:end, 2:end-1]) / (2*dx)
        Vy[2:end-1, 2:end-1]  .-= 0.5/dy .* (P[2:end-1, 3:end] .- P[2:end-1, 1:end-2]) # (diff(P, dims=2)[2:end-1, 1:end-1] + diff(P, dims=2)[2:end-1, 2:end]) / (2*dy)

        # TODO: Boundaries
        bc_noflux!(Vx) # ?
        bc_noflux!(Vy) # ?

        # Density advection
        C_advected   = stams_method(C, Vx, Vy, dt, dx, dy, nx, ny, Lx, Ly, xc, yc)

        # Temperature advection
        T_advected   = stams_method(T, Vx, Vy, dt, dx, dy, nx, ny, Lx, Ly, xc, yc)

        C = C_advected
        # TODO: boundaries on C
        # C[nx÷2-del:nx÷2+del, 1:4] .= 1.0

        T = T_advected
        # TODO: boundaries on T
        T[:,1  ] .=  0.8
        T[:,end] .= -0.8

        # Visualization
        if it % nout == 0
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:inferno#= , title="time = $(round(it*dt, sigdigits=3))" =#)
            plot(heatmap(xc, yc, C', title="C"; opts...), heatmap(xc, yc, P', title="P"; opts...), heatmap(xc, yc, T', title="T"; opts...), heatmap(xc, yc, Vx', title="Vx"; opts...), heatmap(xc, yc, Vy', title="Vy"; opts...))
            frame(anim)
        end
    end

    # Save .gif
    gif(anim, "./dif_$(Base.time()).gif", fps=15)

end

smoke_2D()