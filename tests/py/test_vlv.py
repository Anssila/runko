import unittest
import runko
import numpy as np
import itertools
from scipy.special import kn

def basic_config():
    config = runko.Configuration(None)
    config.Nx = 1
    config.Ny = 1
    config.Nz = 1
    config.NxMesh = 3
    config.NyMesh = 3
    config.NzMesh = 3
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

        # test that fluid accumulates at the corner for very large (negative) acceleration
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
        config.NxMesh = 5
        config.NyMesh = 5
        config.NzMesh = 5
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

    def test_tile_translate(self):
        config = basic_config()
        config.Nvx = 5
        config.Nvy = 5
        config.Nvz = 5
        config.NzMesh = 4
        config.NxMesh = 3
        config.NyMesh = 3
        config.inftyx = 10

        tile = create_tile(config)
        v_init = lambda x, y, z : 0 if x != 0 else 0 if y !=0 else 0 if z!= 5 else 1

        tile.SetVelDistribution(1,1,0, v_init)
        tot = sum(sum(sum(tile.GetVelDistribution(1,1,0))))
        self.assertAlmostEqual(tot, 1)

        for i in range(1,4):
            v_init = lambda x, y, z : 0 
            tile.SetVelDistribution(1,1,i, v_init)
            tot = sum(sum(sum(tile.GetVelDistribution(1,1,i))))
            self.assertAlmostEqual(tot, 0)

        tile.Translate()
        tile.CleanUp()

        tot = sum(sum(sum(tile.GetVelDistribution(1,1,0))))
        self.assertLess(tot, 1)
        tot = sum(sum(sum(tile.GetVelDistribution(1,1,1))))
        self.assertGreater(tot, 0)
        tot = sum(sum(sum(tile.GetVelDistribution(1,1,2))))
        self.assertAlmostEqual(tot, 0)
        tot = sum(sum(sum(tile.GetVelDistribution(1,1,3))))
        self.assertAlmostEqual(tot, 0)

    def test_tile_debug_bc(self):
        # Test that fluid is conserved while translating under periodic (debug) boundary conditions
        config = basic_config()
        config.Nvx = 5
        config.Nvy = 5
        config.Nvz = 5
        config.NzMesh = 4
        config.NxMesh = 3
        config.NyMesh = 3

        tile = create_tile(config)
        v_init = lambda x, y, z : 7 * x + 5 * y + 2 * z + 14 

        for i in range(config.NzMesh):
            tile.SetVelDistribution(1,1,i, v_init)
            tot = sum(sum(sum(tile.GetVelDistribution(1,1,i))))
            self.assertAlmostEqual(tot, 125*14)
        tot = 0
        for i in range(config.NzMesh):
            tot += sum(sum(sum(tile.GetVelDistribution(1,1,i))))
        self.assertAlmostEqual(tot, config.NzMesh*125*14)

        tile.Translate()
        tile.DebugBC()
        tile.CleanUp()

        tot = 0
        for i in range(config.NzMesh):
            tot += sum(sum(sum(tile.GetVelDistribution(1,1,i))))
        self.assertAlmostEqual(tot, config.NzMesh*125*14)

    def test_moment_calculation(self):
        # Test that moments of the velocity distribution are calculated correctly
        dU = 0.2
        k_B = 0.1
        T = 1.0
        m = 1.0
        theta = k_B*T/m
        config = basic_config()
        config.inftyx = None
        config.deltaUx = dU
        tile = create_tile(config)

        def maxwell_juttner(vx,vy,vz, theta, m):
            return 1.0/(4.0*np.pi*theta*kn(2,1.0/theta))*np.exp(-np.sqrt(1.0+vx**2+vy**2+vz**2)/theta)/m**3

        v_init = lambda x, y, z : maxwell_juttner(x,y,z, theta, m)
        tile.SetVelDistribution(1,1,1, v_init)
        grid = tile.GetVelDistribution(1,1,1)
        tot = sum(sum(sum(grid))) * dU**3

        # Test that number density is calculated correctly
        n_lambda = lambda x, y, z, gamma : 1.0
        moment0 = tile.CalculateMoment(1,1,1,n_lambda)
        self.assertAlmostEqual(tot/moment0, 1.0, 5)

        # Create tile in a way to have a good distribution for temperature calculation
        config.inftyx = 10.0
        config.deltaUx = None
        config.Nvx = 50
        config.Nvy = 50
        config.Nvz = 50
        tile = create_tile(config)
        tile.SetVelDistribution(1,1,1,v_init)
        moment0_0 = tile.CalculateMoment(1,1,1,n_lambda)

        # Test that the bulk velocity is zero in all directions
        vx_lambda = lambda x, y, z, gamma : x
        vy_lambda = lambda x, y, z, gamma : y
        vz_lambda = lambda x, y, z, gamma : z
        moment1x = tile.CalculateMoment(1,1,1,vx_lambda)
        moment1y = tile.CalculateMoment(1,1,1,vy_lambda)
        moment1z = tile.CalculateMoment(1,1,1,vz_lambda)
        self.assertAlmostEqual(moment1x, 0.0, 1)
        self.assertAlmostEqual(moment1y, 0.0, 1)
        self.assertAlmostEqual(moment1z, 0.0, 1)

        # Test (non-relativistic) temperature calculation using bulk velocity
        t_lambda = lambda x, y, z, gamma : ((x-moment1x)**2 + (y-moment1y)**2 + (z-moment1z)**2)
        temperature = tile.CalculateMoment(1,1,1,t_lambda) * m / (3*moment0_0*k_B)
        self.assertAlmostEqual(temperature, T, 0)

        # Test that bulk velocity is non-zero after acceleration
        tile.DebugAccelerate(1,1,1, 5.0, -2.0, 7.0, 1.0)
        moment0 = tile.CalculateMoment(1,1,1,n_lambda)
        self.assertAlmostEqual(moment0_0/moment0, 1.0, 5)
        moment1x = tile.CalculateMoment(1,1,1,vx_lambda)
        moment1y = tile.CalculateMoment(1,1,1,vy_lambda)
        moment1z = tile.CalculateMoment(1,1,1,vz_lambda)

        self.assertAlmostEqual(moment1x, 5.0,2)
        self.assertAlmostEqual(moment1y, -2.0,2)
        self.assertAlmostEqual(moment1z, 7.0,2)

if __name__ == "__main__":
    unittest.main()


