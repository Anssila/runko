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

class vlv_tile_general(unittest.TestCase):

    def test_tile_empty(self):
        config = basic_config()
        tile = create_tile(config)

        for x,y,z in itertools.product(range(config.n_cells_per_tile[0]), range(config.n_cells_per_tile[1]), range(config.n_cells_per_tile[2])):

            grid = tile.GetVelDistribution(x,y,z,0)

            self.assertEqual(3, len(grid))
            self.assertEqual(3, len(grid[0]))
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

        correct = [[[ 0.,  0.,  0.], \
                    [ 0.,  0.,  0.], \
                    [ 0.,  0.,  0.]],\
                   [[ 0.,  0.,  0.], \
                    [-3.,  0.,  3.], \
                    [ 0.,  0.,  0.]],\
                   [[ 0.,  0.,  0.], \
                    [ 0.,  0.,  0.], \
                    [ 0.,  0.,  0.]]]


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
        self.assertEqual(tot, 70)

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

        self.assertAlmostEqual(tot/tot3, 1,6)

        # test that fluid accumulates at the corner for very large accelerations
        tile.DebugAccelerate(0,0,0,10.0,10.0,10.0)
        grid = tile.GetVelDistribution(0,0,0,0)
        tot4 = sum(sum(sum(grid)))

        self.assertAlmostEqual(tot/tot4, 1, 6)
        self.assertAlmostEqual(grid[1][1][4]/tot,1, 6)

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
        self.assertAlmostEqual(grid[1][1][0]/tot, 1, 6)

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
            self.assertEqual(tot, 5 * (14+x_+y_+z_))

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
        config.v_grid_extents = [3,3,5]
        config.n_cells_per_tile = [3,3,5]
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

    # def test_tile_debug_bc(self):
    #     # Test that fluid is conserved while translating under periodic (debug) boundary conditions
    #     config = basic_config()
    #     config.v_grid_extents = [5,5,5]
    #     config.n_cells_per_tile = [4,3,3]

    #     tile = create_tile(config)
    #     v_init = lambda x, y, z : 7 * x + 5 * y + 2 * z + 14 

    #     for i in range(config.n_cells_per_tile[2]):
    #         tile.SetVelDistribution(1,1,i, v_init,0)
    #         tot = sum(sum(sum(tile.GetVelDistribution(1,1,i,0))))
    #         self.assertAlmostEqual(tot, 125*14)
    #     tot = 0
    #     for i in range(config.n_cells_per_tile[2]):
    #         tot += sum(sum(sum(tile.GetVelDistribution(1,1,i,0))))
    #     self.assertAlmostEqual(tot, config.n_cells_per_tile[2]*125*14)

    #     tile.Translate()
    #     tile.DebugBC()
    #     tile.CleanUp()

    #     tot = 0
    #     for i in range(config.n_cells_per_tile[2]):
    #         tot += sum(sum(sum(tile.GetVelDistribution(1,1,i,0))))
    #     self.assertAlmostEqual(tot, config.n_cells_per_tile[2]*125*14)

    # def test_set_vlv(self):
    #     config = basic_config()
    #     config.n_cells_per_tile = [5,5,5]
    #     config.v_grid_extents = [11,11,11]
    #     config.u_max = None
    #     config.u_res = [0.1,0.1,0.1]
    #     tile = create_tile(config)

    #     vlv_init = lambda x, y, z, ux, uy, uz : np.exp(-(ux-x/15)**2 - (uy-y/15)**2 - (uz-z/15)**2)

    #     tile.set_vlv(vlv_init,0)

    #     for x_,y_,z_ in itertools.product(range(config.n_cells_per_tile[0]), range(config.n_cells_per_tile[1]), range(config.n_cells_per_tile[2])):
    #         grid = tile.GetVelDistribution(x_,y_,z_,0)
    #         for ux, uy, uz in itertools.product(range(config.v_grid_extents[0]), range(config.v_grid_extents[1]), range(config.v_grid_extents[2])):
    #             self.assertAlmostEqual(grid[ux][uy][uz], vlv_init( \
    #                 x_+0.5, y_+0.5, z_+0.5, \
    #                 ux/10.0-0.5, uy/10.0-0.5, uz/10.0-0.5  \
    #             ))

    # def test_snapshot(self):
    #     config = basic_config()
    #     config.n_cells_per_tile = [4,6,8]
    #     config.v_grid_extents = [3,5,7]
    #     config.u_max = None
    #     config.u_res = [0.1,0.1,0.1]
    #     tile = create_tile(config)

    #     vlv_init = lambda x, y, z, ux, uy, uz : x + 2*y + 3*z + 4*ux + 5*uy + 6*uz

    #     tile.set_vlv(vlv_init,0)

    #     snapshot = tile.get_vlv_snapshot(0)
    #     for x_,y_,z_,ux,uy,uz in itertools.product(\
    #         range(config.n_cells_per_tile[0]), range(config.n_cells_per_tile[1]), range(config.n_cells_per_tile[2]),\
    #         range(config.v_grid_extents[0]), range(config.v_grid_extents[1]), range(config.v_grid_extents[2])):
    #         self.assertAlmostEqual(snapshot[x_][y_][z_][ux][uy][uz], vlv_init( \
    #             x_+0.5, y_+0.5, z_+0.5, \
    #             (ux-1)*0.1, (uy-2)*0.1, (uz-3)*0.1  \
    #         ),5)

if __name__ == "__main__":
    unittest.main()