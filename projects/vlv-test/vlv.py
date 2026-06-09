import runko
import matplotlib.pyplot as plt
import numpy

if __name__ == "__main__":
    config = runko.Configuration(None)
    config.Nx = 1
    config.Ny = 1
    config.Nz = 1
    config.NxMesh = 5
    config.NyMesh = 5
    config.NzMesh = 5
    config.Nvx = 20
    config.Nvy = 20
    config.Nvz = 20
    config.xmin = 0
    config.ymin = 0
    config.zmin = 0
    config.cfl = 1
    config.field_propagator = "FDTD2"
    # config.deltaUx = 1
    # config.deltaUy = 0.2
    # config.deltaUz = 0.4
    config.inftyx = 5.0
    # config.inftyy = 2.0
    # config.inftyz = 3.0
    
    
    tile_grid_idx = (0,0,0)
    
    tile = runko.vlv.threeD.Tile(tile_grid_idx, config)
    v_init = lambda x,y,z : 7*x + 5*y + 2*z + 70
    tile.SetVelDistribution(v_init)
    grid = tile.GetVelDistribution()
    
    # print(grid)
    print()
    print(f"Total fluid: {sum(sum(sum(grid)))}")
    print()
    tile.DebugAccelerate(0.0,0.3,0.8,1.0)
    grid = tile.GetVelDistribution()
    # print(grid)
    print()    
    print(f"Total fluid: {sum(sum(sum(grid)))}")
    
    tile.DebugAccelerate(0.0,-0.3,-0.8, 1.0)
    for i in range(100):
        
        tile.DebugAccelerate(0.0,0.3,0.8,1.0)
        tile.DebugAccelerate(0.5, -0.3, -0.8, 1.0)
        tile.DebugAccelerate(-0.4, 0.24, 0.45, 1.0)
        tile.DebugAccelerate(0.23, -1.16, -0.11, 1.0)
        tile.DebugAccelerate(-0.33, 0.92, -0.34, 1.0)
        grid = tile.GetVelDistribution()
        print(f"Total fluid: {sum(sum(sum(grid)))}")
    numpy.set_printoptions(linewidth=200)
    print(grid)
    print()    
    print(f"Total fluid: {sum(sum(sum(grid)))}")
    tile.DebugAccelerate(-1.0,-1.0,-1.0,10.0)
    grid = tile.GetVelDistribution()
    print(grid)
    print(f"Total fluid: {sum(sum(sum(grid)))}")
    
    # for i in range(len(grid)):
    #     for j in range(len(grid[0])):
    #         print(grid[i][j][0])
    
    