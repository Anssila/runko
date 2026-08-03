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

class vlv_tile_current_deposition(unittest.TestCase):

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
            return 1.0/(4.0*np.pi*theta*kn(2,1.0/theta))*np.exp(-np.sqrt(1.0+vz**2)/theta)/m**3

        v_init = lambda x, y, z : maxwell_juttner(x,y,z, theta, m)
        tile.SetVelDistribution(1,1,1, v_init,0)
        grid = tile.GetVelDistribution(1,1,1,0)
        tot = sum(sum(sum(grid))) * dU

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
        # t_lambda = lambda x, y, z, gamma : ((z-moment1z)**2)
        # temperature = tile.CalculateMoment(1,1,1,t_lambda,0) * m / (3*moment0_0*k_B)
        # self.assertAlmostEqual(temperature, T, 0)

        # Test that bulk velocity is non-zero after acceleration
        tile.DebugAccelerate(1,1,1, 5.0, -2.0, 7.0)
        moment0 = tile.CalculateMoment(1,1,1,n_lambda,0)
        self.assertAlmostEqual(moment0_0/moment0, 1.0, 5)
        moment1x = tile.CalculateMoment(1,1,1,vx_lambda,0)
        moment1y = tile.CalculateMoment(1,1,1,vy_lambda,0)
        moment1z = tile.CalculateMoment(1,1,1,vz_lambda,0)

        self.assertAlmostEqual(moment1x, 0.0,2)
        self.assertAlmostEqual(moment1y, 0.0,2)
        self.assertAlmostEqual(moment1z/moment0, 7.0,2)

    def test_current_deposition(self):
        k_B = 0.1
        T = 0.1
        m = 1.0
        v_0 = np.sqrt(2*k_B*T/m)
        config = basic_config()
        config.u_max = [20*v_0,20*v_0,20*v_0]
        config.u_res = None
        config.v_grid_extents = [50,50,50]
        config.q0 = 0.3
        config.q1 = -0.2
        config.m0 = 1.0
        config.m1 = 1.0
        config.cfl = 0.5
        tile = create_tile(config)

        def maxwell_distr(vx, vy, vz, v_0):# reference velocity = sqrt((2*k*T)/m) (m is mass, k is boltzmann const, T is temperature)
            return (np.pi*v_0**2)**(-0.5) * np.exp(-(vx**2+vy**2+vz**2)/v_0**2)

        v_init = lambda x, y, z : maxwell_distr(x,y,z, v_0)
        tile.SetVelDistribution(1,1,1, v_init,0)
        tile.SetVelDistribution(1,1,1, v_init,1)
        n_lambda = lambda x, y, z, gamma : 1.0
        moment0 = tile.CalculateMoment(1,1,1,n_lambda,0)
        moment1 = tile.CalculateMoment(1,1,1,n_lambda,1)

        # Make sure that the fluid is initialized correctly
        self.assertAlmostEqual(moment0, 1.0,5)
        self.assertAlmostEqual(moment1, 1.0,5)

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
        self.assertAlmostEqual(E0x[1][1][1], 0.0)
        self.assertAlmostEqual(E0y[1][1][1], 0.0)
        self.assertAlmostEqual(E0z[1][1][1], 0.0)

        tile.DebugAccelerate(1,1,1,v_0, -2*v_0, 1.5*v_0)
        tile.deposit_current()
        tile.add_current()

        (E0x, E0y, E0z), (B0x, B0y, B0z), (J0x, J0y, J0z) = tile.get_EBJ()
        self.assertAlmostEqual(J0x[1][1][1], 0.0,2)
        self.assertAlmostEqual(J0y[1][1][1], 0.0,2)
        self.assertAlmostEqual(J0z[1][1][1], 1.5*v_0 * (config.q0**2+config.q1**2),2)
        self.assertAlmostEqual(E0x[1][1][1], 0.0,2)
        self.assertAlmostEqual(E0y[1][1][1], 0.0,2)
        self.assertAlmostEqual(E0z[1][1][1], -1.5*v_0 * (config.q0**2+config.q1**2),2)

if __name__ == "__main__":
    unittest.main()