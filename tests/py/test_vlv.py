import unittest
import runko
import numpy as np
import itertools
from scipy.special import kn

def basic_config():
    config = runko.Configuration(None)
    config.n_tiles = [1,1,1]
    config.n_cells_per_tile = [3,3,3]
    config.v_grid_extents = [10,10,10]
    config.xmin = 0
    config.ymin = 0
    config.zmin = 0
    config.cfl = 1
    config.field_propagator = "fdtd2"
    config.u_max = [1.0,1.0,1.0]
    config.q0 = 1.0
    config.m0 = 1.0
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

        for x,y,z in itertools.product(range(config.n_cells_per_tile[0]), range(config.n_cells_per_tile[1]), range(config.n_cells_per_tile[2])):

            grid = tile.GetVelDistribution(x,y,z,0)

            self.assertEqual(10, len(grid))
            self.assertEqual(10, len(grid[0]))
            self.assertEqual(10, len(grid[0][0]))

            for slice in grid:
                for row in slice:
                    for value in row:
                        self.assertEqual(0, value)

    def test_tile_set_data(self):
        config = basic_config()
        config.v_grid_extents = [3,3,3]
        config.u_max = [1.0,2.0,3.0]
        tile = create_tile(config)

        v_init = lambda x, y, z: x + y + z

        tile.SetVelDistribution(0,0,0,v_init,0)
        grid = tile.GetVelDistribution(0,0,0,0)

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
        config.v_grid_extents = [5,5,5]
        tile = create_tile(config)

        v_init = lambda x, y, z : 7 * x + 5 * y + 2 * z + 14

        tile.SetVelDistribution(0,0,0,v_init,0)

        grid = tile.GetVelDistribution(0,0,0,0)

        tot = sum(sum(sum(grid))) # total fluid in the grid

        # make sure that initialization works
        self.assertEqual(tot, 1750)

        tile.DebugAccelerate(0,0,0,0.0,0.3,0.8)

        grid = tile.GetVelDistribution(0,0,0,0)
        tot2 = sum(sum(sum(grid))) 

        # test that fluid is conserved
        self.assertAlmostEqual(tot/tot2, 1)

        # test that fluid has moved
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                self.assertEqual(grid[i][j][0],0)

        # test fluid conservation for many shifts
        tile.DebugAccelerate(0,0,0,0.0,-0.3,-0.8)
        for i in range(100):
            tile.DebugAccelerate(0,0,0,0.0,0.3,0.8)
            tile.DebugAccelerate(0,0,0,0.5, -0.3, -0.8)
            tile.DebugAccelerate(0,0,0,-0.4, 0.24, 0.45)
            tile.DebugAccelerate(0,0,0,0.23, -1.16, -0.11)
            tile.DebugAccelerate(0,0,0,-0.33, 0.92, -0.34)

        grid = tile.GetVelDistribution(0,0,0,0)
        tot3 = sum(sum(sum(grid)))

        self.assertAlmostEqual(tot/tot3, 1)

        # test that fluid accumulates at the corner for very large accelerations
        tile.DebugAccelerate(0,0,0,10.0,10.0,10.0)
        grid = tile.GetVelDistribution(0,0,0,0)
        tot4 = sum(sum(sum(grid)))

        self.assertAlmostEqual(tot/tot4, 1, 6)
        self.assertAlmostEqual(grid[4][4][4]/tot,1, 6)

        # now test with a larger grid
        config = basic_config()
        config.v_grid_extents = [20,20,20]
        config.u_max = [5.0,5.0,5.0]
        tile = create_tile(config)

        v_init = lambda x, y, z : 7 * x + 5 * y + 2 * z + 70

        tile.SetVelDistribution(0,0,0,v_init,0)

        grid = tile.GetVelDistribution(0,0,0,0)

        tot = sum(sum(sum(grid))) # total fluid in the grid

        tile.DebugAccelerate(0,0,0,0.0,0.3,0.8)

        grid = tile.GetVelDistribution(0,0,0,0)
        tot2 = sum(sum(sum(grid))) 

        # test that fluid is conserved
        self.assertAlmostEqual(tot/tot2, 1)

        # test that fluid has moved
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                self.assertEqual(grid[i][j][0],0)

        # test fluid conservation for many shifts
        tile.DebugAccelerate(0,0,0,0.0,-0.3,-0.8)
        for i in range(100):
            tile.DebugAccelerate(0,0,0,0.0,0.3,0.8)
            tile.DebugAccelerate(0,0,0,0.5, -0.3, -0.8)
            tile.DebugAccelerate(0,0,0,-0.4, 0.24, 0.45)
            tile.DebugAccelerate(0,0,0,0.23, -1.16, -0.11)
            tile.DebugAccelerate(0,0,0,-0.33, 0.92, -0.34)

        grid = tile.GetVelDistribution(0,0,0,0)
        tot3 = sum(sum(sum(grid)))

        self.assertAlmostEqual(tot/tot3, 1)

        # test that fluid accumulates at the corner for very large (negative) acceleration
        tile.DebugAccelerate(0,0,0,-10.0,-10.0,-10.0)
        grid = tile.GetVelDistribution(0,0,0,0)
        tot4 = sum(sum(sum(grid)))
        self.assertAlmostEqual(tot/tot4, 1, 6)
        self.assertAlmostEqual(grid[0][0][0]/tot, 1, 6)

    def test_tile_all_cells(self):
        config = basic_config()
        config.v_grid_extents = [5,5,5]
        config.n_cells_per_tile = [5,5,5]
        tile = create_tile(config)

        for x_,y_,z_ in itertools.product(range(config.n_cells_per_tile[0]), range(config.n_cells_per_tile[1]), range(config.n_cells_per_tile[2])):
            v_init = lambda x, y, z : 7 * x + 5 * y + 2 * z + 14 + x_ + y_ + z_ 

            tile.SetVelDistribution(x_,y_,z_,v_init,0)

            grid = tile.GetVelDistribution(x_,y_,z_,0)

            tot = sum(sum(sum(grid))) # total fluid in the grid

            # make sure that initialization works
            self.assertEqual(tot, 125 * (14+x_+y_+z_))

            tile.DebugAccelerate(x_,y_,z_,0.0,0.3,0.8)

            grid = tile.GetVelDistribution(x_,y_,z_,0)
            tot2 = sum(sum(sum(grid))) 

            # test that fluid is conserved
            self.assertAlmostEqual(tot/tot2, 1)

            # test that fluid has moved
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    self.assertEqual(grid[i][j][0],0)

    def test_tile_translate(self):
        config = basic_config()
        config.v_grid_extents = [5,5,5]
        config.n_cells_per_tile = [4,3,3]
        config.u_max = [10.0,10.0,10.0]

        tile = create_tile(config)
        v_init = lambda x, y, z : 0 if x != 0 else 0 if y !=0 else 0 if z!= 5 else 1

        tile.SetVelDistribution(1,1,0, v_init,0)
        tot = sum(sum(sum(tile.GetVelDistribution(1,1,0,0))))
        self.assertAlmostEqual(tot, 1)

        for i in range(1,4):
            v_init = lambda x, y, z : 0 
            tile.SetVelDistribution(1,1,i, v_init,0)
            tot = sum(sum(sum(tile.GetVelDistribution(1,1,i,0))))
            self.assertAlmostEqual(tot, 0)

        tile.Translate()
        tile.CleanUp()

        tot = sum(sum(sum(tile.GetVelDistribution(1,1,0,0))))
        self.assertLess(tot, 1)
        tot = sum(sum(sum(tile.GetVelDistribution(1,1,1,0))))
        self.assertGreater(tot, 0)
        tot = sum(sum(sum(tile.GetVelDistribution(1,1,2,0))))
        self.assertAlmostEqual(tot, 0)
        tot = sum(sum(sum(tile.GetVelDistribution(1,1,3,0))))
        self.assertAlmostEqual(tot, 0)

    def test_tile_debug_bc(self):
        # Test that fluid is conserved while translating under periodic (debug) boundary conditions
        config = basic_config()
        config.v_grid_extents = [5,5,5]
        config.n_cells_per_tile = [4,3,3]

        tile = create_tile(config)
        v_init = lambda x, y, z : 7 * x + 5 * y + 2 * z + 14 

        for i in range(config.n_cells_per_tile[2]):
            tile.SetVelDistribution(1,1,i, v_init,0)
            tot = sum(sum(sum(tile.GetVelDistribution(1,1,i,0))))
            self.assertAlmostEqual(tot, 125*14)
        tot = 0
        for i in range(config.n_cells_per_tile[2]):
            tot += sum(sum(sum(tile.GetVelDistribution(1,1,i,0))))
        self.assertAlmostEqual(tot, config.n_cells_per_tile[2]*125*14)

        tile.Translate()
        tile.DebugBC()
        tile.CleanUp()

        tot = 0
        for i in range(config.n_cells_per_tile[2]):
            tot += sum(sum(sum(tile.GetVelDistribution(1,1,i,0))))
        self.assertAlmostEqual(tot, config.n_cells_per_tile[2]*125*14)

    def test_moment_calculation(self):
        # Test that moments of the velocity distribution are calculated correctly
        dU = 0.2
        k_B = 0.1
        T = 1.0
        m = 1.0
        theta = k_B*T/m
        config = basic_config()
        config.u_max = None
        config.u_res = [dU,dU,dU]
        tile = create_tile(config)

        def maxwell_juttner(vx,vy,vz, theta, m):
            return 1.0/(4.0*np.pi*theta*kn(2,1.0/theta))*np.exp(-np.sqrt(1.0+vx**2+vy**2+vz**2)/theta)/m**3

        v_init = lambda x, y, z : maxwell_juttner(x,y,z, theta, m)
        tile.SetVelDistribution(1,1,1, v_init,0)
        grid = tile.GetVelDistribution(1,1,1,0)
        tot = sum(sum(sum(grid))) * dU**3

        # Test that number density is calculated correctly
        n_lambda = lambda x, y, z, gamma : 1.0
        moment0 = tile.CalculateMoment(1,1,1,n_lambda,0)
        self.assertAlmostEqual(tot/moment0, 1.0, 5)

        # Create tile in a way to have a good distribution for temperature calculation
        config.u_max = [10.0, 10.0, 10.0]
        config.u_res = None
        config.v_grid_extents = [50,50,50]
        tile = create_tile(config)
        tile.SetVelDistribution(1,1,1,v_init,0)
        moment0_0 = tile.CalculateMoment(1,1,1,n_lambda,0)

        # Test that the bulk velocity is zero in all directions
        vx_lambda = lambda x, y, z, gamma : x
        vy_lambda = lambda x, y, z, gamma : y
        vz_lambda = lambda x, y, z, gamma : z
        moment1x = tile.CalculateMoment(1,1,1,vx_lambda,0)
        moment1y = tile.CalculateMoment(1,1,1,vy_lambda,0)
        moment1z = tile.CalculateMoment(1,1,1,vz_lambda,0)
        self.assertAlmostEqual(moment1x, 0.0, 1)
        self.assertAlmostEqual(moment1y, 0.0, 1)
        self.assertAlmostEqual(moment1z, 0.0, 1)

        # Test (non-relativistic) temperature calculation using bulk velocity
        t_lambda = lambda x, y, z, gamma : ((x-moment1x)**2 + (y-moment1y)**2 + (z-moment1z)**2)
        temperature = tile.CalculateMoment(1,1,1,t_lambda,0) * m / (3*moment0_0*k_B)
        self.assertAlmostEqual(temperature, T, 0)

        # Test that bulk velocity is non-zero after acceleration
        tile.DebugAccelerate(1,1,1, 5.0, -2.0, 7.0)
        moment0 = tile.CalculateMoment(1,1,1,n_lambda,0)
        self.assertAlmostEqual(moment0_0/moment0, 1.0, 5)
        moment1x = tile.CalculateMoment(1,1,1,vx_lambda,0)
        moment1y = tile.CalculateMoment(1,1,1,vy_lambda,0)
        moment1z = tile.CalculateMoment(1,1,1,vz_lambda,0)

        self.assertAlmostEqual(moment1x, 5.0,2)
        self.assertAlmostEqual(moment1y, -2.0,2)
        self.assertAlmostEqual(moment1z, 7.0,2)

    def test_current_deposition(self):
        dU = 0.0002
        k_B = 0.1
        T = 0.00001
        m = 1.0
        v_0 = np.sqrt(2*k_B*T/m)
        config = basic_config()
        config.u_max = None
        config.u_res = [dU,dU,dU]
        config.v_grid_extents = [50,50,50]
        config.q0 = 3.0
        config.cfl = 0.5
        tile = create_tile(config)

        def maxwell_distr(vx, vy, vz, v_0):# reference velocity = sqrt((2*k*T)/m) (m is mass, k is boltzmann const, T is temperature)
            return (np.pi*v_0**2)**(-1.5) * np.exp(-(vx**2+vy**2+vz**2)/v_0**2)

        v_init = lambda x, y, z : maxwell_distr(x,y,z, v_0)
        tile.SetVelDistribution(1,1,1, v_init,0)
        n_lambda = lambda x, y, z, gamma : 1.0
        moment0 = tile.CalculateMoment(1,1,1,n_lambda,0)

        # Make sure that the fluid is initialized correctly
        self.assertAlmostEqual(moment0, 1.0, 1)

        # Test that J is zero everywhere before the deposition
        (E0x, E0y, E0z), (B0x, B0y, B0z), (J0x, J0y, J0z) = tile.get_EBJ()
        self.assertTrue(np.all(J0x == 0))
        self.assertTrue(np.all(J0y == 0))
        self.assertTrue(np.all(J0z == 0))

        # Test that J is non zero after the deposition
        Exinit = lambda x, y, z : x - x
        Eyinit = lambda x, y, z : y - y
        Ezinit = lambda x, y, z : z - z
        Bxinit = lambda x, y, z : 2 * x
        Byinit = lambda x, y, z : 2 * y
        Bzinit = lambda x, y, z : 2 * z
        Jxinit = lambda x, y, z : 3 * x
        Jyinit = lambda x, y, z : 3 * y
        Jzinit = lambda x, y, z : 3 * z

        tile.batch_set_EBJ(Exinit, Eyinit, Ezinit,
                           Bxinit, Byinit, Bzinit,
                           Jxinit, Jyinit, Jzinit)
        tile.deposit_current()
        (E0x, E0y, E0z), (B0x, B0y, B0z), (J0x, J0y, J0z) = tile.get_EBJ()
        self.assertAlmostEqual(J0x[1][1][1], 0.0)
        self.assertAlmostEqual(J0y[1][1][1], 0.0)
        self.assertAlmostEqual(J0z[1][1][1], 0.0)

        tile.DebugAccelerate(1,1,1,dU, -2*dU, 3*dU)
        tile.deposit_current()
        tile.add_current()

        (E0x, E0y, E0z), (B0x, B0y, B0z), (J0x, J0y, J0z) = tile.get_EBJ()
        self.assertAlmostEqual(J0x[1][1][1], dU * config.q0 * config.cfl)
        self.assertAlmostEqual(J0y[1][1][1], -2*dU * config.q0 * config.cfl)
        self.assertAlmostEqual(J0z[1][1][1], 3*dU * config.q0 * config.cfl)


if __name__ == "__main__":
    unittest.main()