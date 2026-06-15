import unittest
import runko
import numpy as np
import itertools

def basic_config():
    config = runko.Configuration(None)
    config.Nx = 1
    config.Ny = 1
    config.Nz = 1
    config.NxMesh = 5
    config.NyMesh = 5
    config.NzMesh = 5
    config.Nvx = 10
    config.Nvy = 10
    config.Nvz = 10
    config.xmin = 0
    config.ymin = 0
    config.zmin = 0
    config.cfl = 1
    config.field_propagator = "FDTD2"
    config.inftyx = 1
    return config
    

def create_tile(config = None):
    if config == None:
        config = basic_config()

    tile_grid_idx = (0,0,0)
    
    tile = runko.vlv.threeD.Tile(tile_grid_idx, config)
    return tile

class vlv_tile(unittest.TestCase):

    def test_tile_empty(self):
        config = basic_config()
        tile = create_tile(config)
        
        for x,y,z in itertools.product(range(config.NxMesh), range(config.NyMesh), range(config.NzMesh)):
            
            grid = tile.GetVelDistribution(x,y,z)

            self.assertEqual(10, len(grid))
            self.assertEqual(10, len(grid[0]))
            self.assertEqual(10, len(grid[0][0]))
            
            for slice in grid:
                for row in slice:
                    for value in row:
                        self.assertEqual(0, value)
                    
    def test_tile_set_data(self):
        config = basic_config()
        config.Nvx = 3
        config.Nvy = 3
        config.Nvz = 3
        config.inftyx = 1
        config.inftyy = 2
        config.inftyz = 3
        tile = create_tile(config)
        
        v_init = lambda x, y, z: x + y + z
        
        tile.SetVelDistribution(0,0,0,v_init)
        grid = tile.GetVelDistribution(0,0,0)
        
        correct = [[[-6., -3.,  0.], \
                    [-4., -1.,  2.], \
                    [-2.,  1.,  4.]],\
                   [[-5., -2.,  1.], \
                    [-3.,  0.,  3.], \
                    [-1.,  2.,  5.]],\
                   [[-4., -1.,  2.], \
                    [-2.,  1.,  4.], \
                    [ 0.,  3.,  6.]]]

        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                for k in range(len(grid[0][0])):
                    self.assertAlmostEqual(correct[i][j][k], grid[i][j][k])
                    
    def test_tile_accelerate(self):
        config = basic_config()
        config.Nvx = 5
        config.Nvy = 5
        config.Nvz = 5
        tile = create_tile(config)
        
        v_init = lambda x, y, z : 7 * x + 5 * y + 2 * z + 14
        
        tile.SetVelDistribution(0,0,0,v_init)
        
        grid = tile.GetVelDistribution(0,0,0)
        
        tot = sum(sum(sum(grid))) # total fluid in the grid
        
        # make sure that initialization works
        self.assertEqual(tot, 1750)
        
        tile.DebugAccelerate(0,0,0,0.0,0.3,0.8,1.0)
        
        grid = tile.GetVelDistribution(0,0,0)
        tot2 = sum(sum(sum(grid))) 
        
        # test that fluid is conserved
        self.assertAlmostEqual(tot/tot2, 1)
        
        # test that fluid has moved
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                self.assertEqual(grid[i][j][0],0)
        
        # test fluid conservation for many shifts
        tile.DebugAccelerate(0,0,0,0.0,-0.3,-0.8, 1.0)
        for i in range(100):
            tile.DebugAccelerate(0,0,0,0.0,0.3,0.8,1.0)
            tile.DebugAccelerate(0,0,0,0.5, -0.3, -0.8, 1.0)
            tile.DebugAccelerate(0,0,0,-0.4, 0.24, 0.45, 1.0)
            tile.DebugAccelerate(0,0,0,0.23, -1.16, -0.11, 1.0)
            tile.DebugAccelerate(0,0,0,-0.33, 0.92, -0.34, 1.0)
        
        grid = tile.GetVelDistribution(0,0,0)
        tot3 = sum(sum(sum(grid)))
        
        self.assertAlmostEqual(tot/tot3, 1)
        
        # test that fluid accumulates at the corner for very large accelerations
        tile.DebugAccelerate(0,0,0,0.01,0.01,0.01,1000.0)
        grid = tile.GetVelDistribution(0,0,0)
        tot4 = sum(sum(sum(grid)))
        
        self.assertAlmostEqual(tot/tot4, 1, 6)
        self.assertAlmostEqual(grid[4][4][4]/tot,1, 6)
        
        # now test with a larger grid
        config = basic_config()
        config.Nvx = 20
        config.Nvy = 20
        config.Nvz = 20
        config.inftyx = 5
        tile = create_tile(config)
        
        v_init = lambda x, y, z : 7 * x + 5 * y + 2 * z + 70
        
        tile.SetVelDistribution(0,0,0,v_init)
        
        grid = tile.GetVelDistribution(0,0,0)
        
        tot = sum(sum(sum(grid))) # total fluid in the grid
        
        tile.DebugAccelerate(0,0,0,0.0,0.3,0.8,1.0)
        
        grid = tile.GetVelDistribution(0,0,0)
        tot2 = sum(sum(sum(grid))) 
        
        # test that fluid is conserved
        self.assertAlmostEqual(tot/tot2, 1)
        
        # test that fluid has moved
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                self.assertEqual(grid[i][j][0],0)
        
        # test fluid conservation for many shifts
        tile.DebugAccelerate(0,0,0,0.0,-0.3,-0.8, 1.0)
        for i in range(100):
            tile.DebugAccelerate(0,0,0,0.0,0.3,0.8,1.0)
            tile.DebugAccelerate(0,0,0,0.5, -0.3, -0.8, 1.0)
            tile.DebugAccelerate(0,0,0,-0.4, 0.24, 0.45, 1.0)
            tile.DebugAccelerate(0,0,0,0.23, -1.16, -0.11, 1.0)
            tile.DebugAccelerate(0,0,0,-0.33, 0.92, -0.34, 1.0)
        
        grid = tile.GetVelDistribution(0,0,0)
        tot3 = sum(sum(sum(grid)))
        
        self.assertAlmostEqual(tot/tot3, 1)
        
        # test that fluid accelerates at the corner for very large (negative) acceleration
        tile.DebugAccelerate(0,0,0,-1.0,-1.0,-1.0,10.0)
        grid = tile.GetVelDistribution(0,0,0)
        tot4 = sum(sum(sum(grid)))
        self.assertAlmostEqual(tot/tot4, 1, 6)
        self.assertAlmostEqual(grid[0][0][0]/tot, 1, 6)
    
    def test_tile_all_cells(self):
        config = basic_config()
        config.Nvx = 5
        config.Nvy = 5
        config.Nvz = 5
        tile = create_tile(config)
        
        for x_,y_,z_ in itertools.product(range(config.NxMesh), range(config.NyMesh), range(config.NzMesh)):
            v_init = lambda x, y, z : 7 * x + 5 * y + 2 * z + 14 + x_ + y_ + z_ 
            
            tile.SetVelDistribution(x_,y_,z_,v_init)
        
            grid = tile.GetVelDistribution(x_,y_,z_)
            
            tot = sum(sum(sum(grid))) # total fluid in the grid
            
            # make sure that initialization works
            self.assertEqual(tot, 125 * (14+x_+y_+z_))
            
            tile.DebugAccelerate(x_,y_,z_,0.0,0.3,0.8,1.0)
            
            grid = tile.GetVelDistribution(x_,y_,z_)
            tot2 = sum(sum(sum(grid))) 
            
            # test that fluid is conserved
            self.assertAlmostEqual(tot/tot2, 1)
            
            # test that fluid has moved
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    self.assertEqual(grid[i][j][0],0)
        



if __name__ == "__main__":
    unittest.main()
