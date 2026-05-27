// Copyright 2026 - 2026, Miro Palmu, Joonas Nättilä and the runko contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <boost/ut.hpp>
#include "runko/mdgrid_common.h"
#include "runko/vlv/vlasov_grid.h"

namespace {

using namespace boost::ut;
[[maybe_unused]]
const suite<"grid testing"> s1 = [] {
  "grid"_test = [] {
    runko::ScalarGrid<float> grid = runko::ScalarGrid<float>(3,4,5);

    expect(true);
  };

  // this test is a direct copy from tyvi's own tests
  "mdgrid staging buffer round trip"_test = [] {
        constexpr auto elem_desc = tyvi::mdgrid_element_descriptor<int>{ .rank = 2, .dim = 3 };

        using mdg = tyvi::mdgrid<elem_desc, std::dextents<std::size_t, 3>>;

        auto grid = mdg(3, 4, 5);

        auto staging_mds_A = grid.staging_mds();
        for (const auto idx : tyvi::sstd::index_space(staging_mds_A)) {
            for (const auto Midx : tyvi::sstd::index_space(staging_mds_A[idx])) {
                staging_mds_A[idx][Midx] = 7;
            }
        }

        const auto w = tyvi::mdgrid_work{};
        w.sync_from_staging(grid).for_each(grid, [](const auto& M) {
            for (const auto Midx : tyvi::sstd::index_space(M)) { M[Midx] = M[Midx] * 3 * 2; }
        });

        w.sync_to_staging(grid).wait();

        auto staging_mds_B = grid.staging_mds();
        for (const auto idx : tyvi::sstd::index_space(staging_mds_B)) {
            for (const auto Midx : tyvi::sstd::index_space(staging_mds_B[idx])) {
                expect(staging_mds_B[idx][Midx] == 42);
            }
        }
    };

    // this test is a direct copy from tyvi's own tests
    "multiple mdgrids in the same kernel with *::for_each_index"_test = [] {
        constexpr auto scalar_desc = tyvi::mdgrid_element_descriptor<int>{ .rank = 0, .dim = 3 };
        constexpr auto vec_desc    = tyvi::mdgrid_element_descriptor<int>{ .rank = 1, .dim = 3 };

        using scalar_mdg = tyvi::mdgrid<scalar_desc, std::dextents<std::size_t, 3>>;
        using vec_mdg    = tyvi::mdgrid<vec_desc, std::dextents<std::size_t, 3>>;

        auto scalar_grid = scalar_mdg(8, 4, 6);
        auto vec_grid    = vec_mdg(8, 4, 6);

        const auto w          = tyvi::mdgrid_work{};
        const auto [w1a, w1b] = w.split<2>();

        {
            const auto smds_scalar = scalar_grid.staging_mds();
            for (const auto idx : tyvi::sstd::index_space(smds_scalar)) {
                smds_scalar[idx][] = static_cast<int>(idx[0]);
            }
        }
        w1a.sync_from_staging(scalar_grid);

        {
            const auto smds_vec = vec_grid.staging_mds();
            for (const auto idx : tyvi::sstd::index_space(smds_vec)) {
                for (const auto jdx : tyvi::sstd::index_space(smds_vec[idx])) {
                    smds_vec[idx][jdx] = static_cast<int>(jdx[0]);
                }
            }
        }
        w1b.sync_from_staging(vec_grid);

        tyvi::when_all(w1a, w1b);

        auto kernelA = [TYVI_CMDS(vec_grid, scalar_grid)](const auto& idx) {
            for (const auto jdx : tyvi::sstd::index_space(vec_grid_mds[idx])) {
                vec_grid_mds[idx][jdx] = vec_grid_mds[idx][jdx] * scalar_grid_mds[idx][];
            }
        };

        w1a.for_each_index(vec_grid, std::move(kernelA));

        auto kernelB = [TYVI_CMDS(vec_grid, scalar_grid)](const auto& idx, const auto& jdx) {
            vec_grid_mds[idx][jdx] = vec_grid_mds[idx][jdx] * scalar_grid_mds[idx][];
        };

        w1a.for_each_index(vec_grid, std::move(kernelB)).sync_to_staging(vec_grid).wait();

        {
            const auto smds_scalar = scalar_grid.staging_mds();
            const auto smds_vec    = vec_grid.staging_mds();

            for (const auto idx : tyvi::sstd::index_space(smds_vec)) {
                const auto s = static_cast<int>(idx[0]);
                expect(smds_scalar[idx][] == s);

                for (const auto jdx : tyvi::sstd::index_space(smds_vec[idx])) {
                    expect(smds_vec[idx][jdx] == s * s * static_cast<int>(jdx[0]));
                }
            }
        }
    };
};

const suite<"dense grid testing"> s2 = [] {

  "dense_grid"_test = [] {
    auto _ = vlv::DenseGrid(5,4,5);
    expect(true); 
  };

  "shift"_test = [] {
    auto g = vlv::DenseGrid(5,4,5);
    g.Shift(0.1f,0.2f,0.3f);
    expect(g.GetDummy() == 0.6f);
  };

  "initialize"_test = [] {
    auto g = vlv::DenseGrid(3,4,5);
    g.InitZero();
    expect(g.GetTotalFluid() == 0.0f);
  };
};

}  // namespace

int
  main(int argc, const char** argv)
{
  return static_cast<int>(cfg<override>.run(run_cfg { .argc = argc, .argv = argv }));
}
