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

class vlv_tile_accelerate(unittest.TestCase):
    def test_accelerate(self):
        dU = 0.001
        k_B = 0.1
        T = 0.00001
        m = 1.0
        v_0 = np.sqrt(2*k_B*T/m)
        config = basic_config()
        config.u_max = None
        config.u_res = [dU,dU,dU]
        config.v_grid_extents = [10,10,10]
        config.n_cells_per_tile =  [9,9,9]
        config.q0 = 3.0
        config.cfl = 0.5
        tile = create_tile(config)

        def maxwell_distr(vx, vy, vz, v_0):# reference velocity = sqrt((2*k*T)/m) (m is mass, k is boltzmann const, T is temperature)
            return (np.pi*v_0**2)**(-0.5) * np.exp(-(vz**2)/v_0**2)

        v_init = lambda x, y, z : maxwell_distr(x,y,z, v_0)
        for x_,y_,z_ in itertools.product(range(1), range(1), range(config.n_cells_per_tile[2])):
            tile.SetVelDistribution(x_,y_,z_, v_init,0)
        n_lambda = lambda x, y, z, gamma : 1.0
        moment0 = tile.CalculateMoment(1,1,1,n_lambda,0)

        # Make sure that the fluid is initialized correctly
        self.assertAlmostEqual(moment0, 1.0, 1)

        # Initialize E-field so that there will be some acceleration
        Exinit = lambda x, y, z : x / 20000
        Eyinit = lambda x, y, z : y / 20000
        Ezinit = lambda x, y, z : z / 20000
        Bxinit = lambda x, y, z : 2 * x
        Byinit = lambda x, y, z : 2 * y
        Bzinit = lambda x, y, z : 2 * z
        Jxinit = lambda x, y, z : 3 * x
        Jyinit = lambda x, y, z : 3 * y
        Jzinit = lambda x, y, z : 3 * z

        tile.batch_set_EBJ(Exinit, Eyinit, Ezinit,
                           Bxinit, Byinit, Bzinit,
                           Jxinit, Jyinit, Jzinit)

        # Test first that J is zero by depositing current
        tile.deposit_current()
        (E0x, E0y, E0z), (B0x, B0y, B0z), (J0x, J0y, J0z) = tile.get_EBJ()
        for x_,y_,z_ in itertools.product(range(config.n_cells_per_tile[0]), range(config.n_cells_per_tile[1]), range(config.n_cells_per_tile[2])):
            self.assertAlmostEqual(J0x[x_][y_][z_], 0.0)
            self.assertAlmostEqual(J0y[x_][y_][z_], 0.0)
            self.assertAlmostEqual(J0z[x_][y_][z_], 0.0)

        # Now accelerate the fluid based on the E-field and deposit the current that has formed
        tile.accelerate()
        tile.deposit_current()

        # Test that the formed current is correct
        (E0x, E0y, E0z), (B0x, B0y, B0z), (J0x, J0y, J0z) = tile.get_EBJ()
        for x_,y_,z_ in itertools.product(range(config.n_cells_per_tile[0]), range(config.n_cells_per_tile[1]), range(config.n_cells_per_tile[2])):
            self.assertAlmostEqual(J0x[x_][y_][z_], 0.0,5)
            self.assertAlmostEqual(J0y[x_][y_][z_], 0.0,5)
            self.assertAlmostEqual(J0z[x_][y_][z_], config.q0**2/config.m0*config.cfl*E0z[x_][y_][z_] if x_ == 1 and y_ == 1 else 0.0,5)

if __name__ == "__main__":
    unittest.main()