// Copyright 2026 - 2026, Miro Palmu, Joonas Nättilä and the runko contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <boost/ut.hpp>
#include "runko/mdgrid_common.h"
#include "runko/vlv/vlasov_grid.h"
#include <sstream>
#include <iostream>
#include <iomanip>
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

static constexpr vlv::VlasovGrid::value_type tolerance = 1.0e-7f;

const suite<"dense grid testing"> s2 = [] {

  "dense_grid"_test = [] {
    auto _ = vlv::DenseGrid(5,4,5);
    expect(true); 
  };

  "initialize"_test = [] {
    auto g = vlv::DenseGrid(3,4,5);
    g.set_u_res({0.1f,0.1f,0.1f});
    // Test zero initialization

    g.InitZero();
    expect(g.DebugGetTotalFluid() == 0.0f);

    // Test delta initialization
    std::array<vlv::VlasovGrid::value_type,3> v = {0.0f, 0.1f, -0.1f};
    auto inds = g.GetIndFromVel(v);
    g.InitDelta(v);
    expect(g.DebugGetTotalFluid() == 1.0f) << "Expected 1, got " << g.DebugGetTotalFluid();
    expect(g.DebugGetFluid(inds) == 1.0f) << "Expected 1, got " << g.DebugGetFluid(inds);
  };

  "shift"_test = [] {
    auto g = vlv::DenseGrid(5,5,5);
    g.set_u_res({0.1f,0.1f,0.1f});

    // First test that zero fluid stays as zero
    g.InitZero(); 
    g.Shift(0.1f,0.2f,0.3f);
    expect(std::abs(static_cast<float>(g.DebugGetTotalFluid())) < tolerance) << "Expected 0, got " << g.DebugGetTotalFluid();

    // Test that a delta distribution conserves fluid under shifting
    g.InitDelta({0.1f,-0.1f,0.2f});
    vlv::VlasovGrid::value_type tot = g.DebugGetTotalFluid();
    expect(tot == 1.0f);
    g.Shift(0.0f, 0.0f, 0.0f);
    expect(std::abs(static_cast<float>(tot - g.DebugGetTotalFluid())) < tolerance) << "Expected " << tot << ", got " << g.DebugGetTotalFluid();
    g.Shift(0.05f, 0.0f, 0.0f);
    expect(std::abs(static_cast<float>(tot - g.DebugGetTotalFluid())) < tolerance) << "Expected " << tot << ", got " << g.DebugGetTotalFluid();
    g.Shift(-0.1f, 0.0f, 0.0f);
    expect(std::abs(static_cast<float>(tot - g.DebugGetTotalFluid())) < tolerance) << "Expected " << tot << ", got " << g.DebugGetTotalFluid();
    g.Shift(0.05f, 0.05f, 0.05f);
    expect(std::abs(static_cast<float>(tot - g.DebugGetTotalFluid())) < tolerance) << "Expected " << tot << ", got " << g.DebugGetTotalFluid();
    g.Shift(1.05f, 1.05f, 1.05f);
    expect(std::abs(static_cast<float>(tot - g.DebugGetTotalFluid())) < tolerance) << "Expected " << tot << ", got " << g.DebugGetTotalFluid();
    expect(std::abs(static_cast<float>(tot - g.DebugGetFluid({4,4,4}))) < tolerance) << "Expected " << tot << ", got " << g.DebugGetFluid({4,4,4});
    g.Shift(-0.2f,-0.2f,-0.2f);
    expect(std::abs(static_cast<float>(tot - g.DebugGetTotalFluid())) < tolerance) << "Expected " << tot << ", got " << g.DebugGetTotalFluid();
    expect(std::abs(static_cast<float>(tot - g.DebugGetFluid({2,2,2}))) < tolerance) << "Expected " << tot << ", got " << g.DebugGetFluid({2,2,2});
    g.Shift(-2.0f, -2.0f, -2.0f);
    expect(std::abs(static_cast<float>(tot - g.DebugGetTotalFluid())) < tolerance) << "Expected " << tot << ", got " << g.DebugGetTotalFluid();
    expect(std::abs(static_cast<float>(tot - g.DebugGetFluid({0,0,0}))) < tolerance) << "Expected " << tot << ", got " << g.DebugGetFluid({0,0,0});


    g.InitDelta({0.0f,0.0f,0.0f});
    g.Shift(0.03f,-0.01f,0.18f);
    expect(std::abs(static_cast<float>(tot - g.DebugGetTotalFluid())) < tolerance) << "Expected " << tot << ", got " << g.DebugGetTotalFluid();
    g.Shift(-0.03f,0.01f,-0.18f);
    expect(std::abs(static_cast<float>(tot - g.DebugGetTotalFluid())) < tolerance) << "Expected " << tot << ", got " << g.DebugGetTotalFluid();
    g.Shift(0.09f,-0.21f,3.1455f);
    expect(std::abs(static_cast<float>(tot - g.DebugGetTotalFluid())) < tolerance) << "Expected " << tot << ", got " << g.DebugGetTotalFluid();
    g.Shift(0.03f,0.37f,0.0f);
    expect(std::abs(static_cast<float>(tot - g.DebugGetTotalFluid())) < tolerance) << "Expected " << tot << ", got " << g.DebugGetTotalFluid();

  };

  "GetVelFromIndex"_test = [] {
    auto g = vlv::DenseGrid(5,6,7);
    g.set_u_max({2.0f,2.0f,2.0f});

    auto approxEq = [] (std::array<vlv::VlasovGrid::value_type,3> a, std::array<vlv::VlasovGrid::value_type,3> b){
        expect(std::abs(static_cast<float>(a[0]-b[0])) < tolerance) << "Expected " << b[0] << ", got " << a[0];
        expect(std::abs(static_cast<float>(a[1]-b[1])) < tolerance) << "Expected " << b[1] << ", got " << a[1];
        expect(std::abs(static_cast<float>(a[2]-b[2])) < tolerance) << "Expected " << b[2] << ", got " << a[2];
    };

    approxEq(g.GetVelFromInd({0,0,0}),{-2.0f,-2.0f,-2.0f});

    approxEq(g.GetVelFromInd({4,5,6}),{2.0f,2.0f,2.0f});

    approxEq(g.GetVelFromInd({2,2,3}),{0.0f,-0.4f,0.0f});
  };

  "GetIndexFromVel"_test = [] {
    auto g = vlv::DenseGrid(5,6,7);
    g.set_u_res({0.1f,0.1f,0.1f});

    expect(g.GetIndFromVel({-0.2f,-0.25f,-0.3f}) == std::array<runko::index_t,3>{0,0,0});

    expect(g.GetIndFromVel({0.2f,0.25f,0.3f}) == std::array<runko::index_t,3>{4,5,6});

    expect(g.GetIndFromVel({0.0f,0.0f,0.0f}) == std::array<runko::index_t,3>{2,2,3});
  };

  "odd_index"_test = [] {
    auto g = vlv::DenseGrid(7,7,7);
    g.set_u_res({0.1f,0.1f,0.1f});

    auto test_inds = std::vector<std::array<runko::index_t,3>>{
        {0,0,0},
        {6,6,6},
        {3,3,3},
        {1,2,3}
    };
    for (auto inds : test_inds){ 
        auto v = g.GetVelFromInd(inds);
        auto t = g.GetIndFromVel(v);
        expect(t == inds) << "Expected {" 
                          << inds[0] << ","
                          << inds[1] << ","
                          << inds[2] << "}, got {"
                          << t[0] << ","
                          << t[1] << ","
                          << t[2] << "}. v is {"
                          << v[0] << ", "
                          << v[1] << ", "
                          << v[2] << "}";
    }

    auto test_vels = std::vector<std::array<vlv::VlasovGrid::value_type,3>>{
        {0.0f,0.0f,0.0f},
        {-0.3f,-0.3f,-0.3f},
        {0.3f,0.3f,0.3f},
        {0.0f,0.1f,0.2f}
    };

    auto approxEq = [] (std::array<vlv::VlasovGrid::value_type,3> a, std::array<vlv::VlasovGrid::value_type,3> b){
        expect(std::abs(static_cast<float>(a[0]-b[0])) < tolerance) << "Expected " << b[0] << ", got " << a[0];
        expect(std::abs(static_cast<float>(a[1]-b[1])) < tolerance) << "Expected " << b[1] << ", got " << a[1];
        expect(std::abs(static_cast<float>(a[2]-b[2])) < tolerance) << "Expected " << b[2] << ", got " << a[2];
    };

    for (auto vel : test_vels){
        approxEq(g.GetVelFromInd(g.GetIndFromVel(vel)),vel);
    }

  };

  "even_index"_test = [] {
    auto g = vlv::DenseGrid(8,8,8);
    g.set_u_res({0.1f,0.1f,0.1f});

    auto test_inds = std::vector<std::array<runko::index_t,3>>{
        {0,0,0},
        {7,7,7},
        {3,3,3},
        {4,4,4},
        {1,2,3}
    };
    for (auto inds : test_inds){ 
        auto v = g.GetVelFromInd(inds);
        auto t = g.GetIndFromVel(v);
        expect(t == inds) << "Expected {" 
                          << inds[0] << ","
                          << inds[1] << ","
                          << inds[2] << "}, got {"
                          << t[0] << ","
                          << t[1] << ","
                          << t[2] << "}. v is {"
                          << v[0] << ", "
                          << v[1] << ", "
                          << v[2] << "}";
    }

    auto test_vels = std::vector<std::array<vlv::VlasovGrid::value_type,3>>{
        {0.05f,0.05f,0.05f},
        {-0.35f,-0.35f,-0.35f},
        {0.35f,0.35f,0.35f},
        {-0.05f,-0.15f,-0.25f}
    };

    auto approxEq = [] (std::array<vlv::VlasovGrid::value_type,3> a, std::array<vlv::VlasovGrid::value_type,3> b){
        expect(std::abs(static_cast<float>(a[0]-b[0])) < tolerance) << "Expected " << b[0] << ", got " << a[0];
        expect(std::abs(static_cast<float>(a[1]-b[1])) < tolerance) << "Expected " << b[1] << ", got " << a[1];
        expect(std::abs(static_cast<float>(a[2]-b[2])) < tolerance) << "Expected " << b[2] << ", got " << a[2];
    };

    for (auto vel : test_vels){
        approxEq(g.GetVelFromInd(g.GetIndFromVel(vel)),vel);
    }

  };

  "translate"_test = [] {
    std::vector<vlv::DenseGrid> grids;
    for (int i = 0; i < 3; i++){
        grids.emplace_back(5,5,5);
        grids[i].set_u_res({0.1f,0.1f,0.1f});
        grids[i].InitDelta({0.2f,-0.1f,0.1f});
        vlv::VlasovGrid::value_type tot = grids[i].DebugGetTotalFluid();
        expect(tot == 1.0f) << "Expected 1.0, got " << tot;  
    }
    auto neighbors = std::vector<vlv::VlasovGrid*>{
        static_cast<vlv::VlasovGrid*>(&grids[0]),
        static_cast<vlv::VlasovGrid*>(&grids[1]),
        static_cast<vlv::VlasovGrid*>(&grids[2])
    };

    // Test that fluid is moved from one grid to the other

    grids[1].TranslateZ(neighbors, 1.0f);
    grids[1].Clean();
    grids[0].Clean();
    grids[2].Clean();
    auto tot1 = grids[0].DebugGetTotalFluid();
    expect(tot1 == 0.0f) << "Expected 0.0, got " << tot1;
    auto tot2 = grids[1].DebugGetTotalFluid();
    expect(tot2 < 1.0f) << "Expected < 1.0, got " << tot2;
    expect(tot2 > 0.0f) << "Expected > 0.0, got " << tot2;
    auto tot3 = grids[2].DebugGetTotalFluid();
    expect(tot3 < 1.0f) << "Expected < 1.0, got " << tot3;
    expect(tot3 > 0.0f) << "Expected > 0.0, got " << tot3;

    expect(tot1+tot2+tot3 == 1.0) << "Expected total to be 1.0, got " << tot1+tot2+tot3 << " from: " << tot1 << " + " << tot2 << " + " << tot3;  

    // Test that fluid is conserved when moving around

    grids[0].InitDelta({0.1f,0.1f,-0.1f});
    grids[1].InitDelta({0.1f,0.1f,-0.1f});
    grids[2].InitDelta({0.1f,0.1f,-0.1f});

    tot1 = grids[0].DebugGetTotalFluid();
    tot2 = grids[1].DebugGetTotalFluid();
    tot3 = grids[2].DebugGetTotalFluid();

    expect(tot1+tot2+tot3 == 3.0) << "Expected total to be 3.0, got " << tot1+tot2+tot3 << " from: " << tot1 << " + " << tot2 << " + " << tot3; 



    // Use periodic boundary conditions
    grids[0].TranslateZ({neighbors[2],neighbors[0],neighbors[1]},1.0f);
    grids[1].TranslateZ({neighbors[0],neighbors[1],neighbors[2]},1.0f);
    grids[2].TranslateZ({neighbors[1],neighbors[2],neighbors[0]},1.0f);

    grids[0].Clean();
    grids[1].Clean();
    grids[2].Clean();

    tot1 = grids[0].DebugGetTotalFluid();
    tot2 = grids[1].DebugGetTotalFluid();
    tot3 = grids[2].DebugGetTotalFluid();

    expect(tot1+tot2+tot3 == 3.0) << "Expected total to be 3.0, got " << tot1+tot2+tot3 << " from: " << tot1 << " + " << tot2 << " + " << tot3; 

    // Test that the fluid is in the correct configuration:
    const vlv::VlasovGrid::value_type correct[5][5][5] = {
        {
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f}
        },
        {
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f}
        },
        {
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f}
        },
        {
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,1.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f}
        },
        {
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f},
            {0.0f,0.0f,0.0f,0.0f,0.0f}
        }
    };
    for (uint i = 0; i < 5; i++)
    for (uint j = 0; j < 5; j++)
    for (uint k = 0; k < 5; k++){
        auto val1 = grids[0].DebugGetFluid({i,j,k});
        auto val2 = grids[1].DebugGetFluid({i,j,k});
        auto val3 = grids[2].DebugGetFluid({i,j,k});
        expect(val1 == correct[i][j][k]) << "Expected " << correct[i][j][k] << ", got " << val1 << " for grid 1 and indices {" << i << ", " << j << ", " << k << "}";
        expect(val2 == correct[i][j][k]) << "Expected " << correct[i][j][k] << ", got " << val2 << " for grid 2 and indices {" << i << ", " << j << ", " << k << "}";
        expect(val3 == correct[i][j][k]) << "Expected " << correct[i][j][k] << ", got " << val3 << " for grid 3 and indices {" << i << ", " << j << ", " << k << "}";
    }
  };

  "moment"_test = [] {
    auto g = vlv::DenseGrid(5,5,5);
    g.set_u_res({0.1f,0.1f,0.1f});
    g.InitDelta({0.0f,0.0f,0.1f});
    expect(g.DebugGetTotalFluid() == 1.0f);
    const auto w = tyvi::mdgrid_work{};
    const auto n = [] ([[maybe_unused]] double x, [[maybe_unused]] double y, [[maybe_unused]] double z, [[maybe_unused]] double gamma){
        return 1.0f;
    };

    auto approxEq = [] (vlv::VlasovGrid::value_type a, vlv::VlasovGrid::value_type b){
        expect(std::abs(static_cast<float>(a-b)) < 1e-8) << "Expected " << std::setprecision(10) << b << ", got " << a;
    };

    approxEq(g.CalculateMoment(w, n), 0.001f);
    const auto u = [] ([[maybe_unused]] double x, [[maybe_unused]] double y, [[maybe_unused]] double z, [[maybe_unused]] double gamma){
        return std::sqrt(x*x+y*y+z*z);
    };

    approxEq(g.CalculateMoment(w, u), 0.0001f);

    const auto v = [] ([[maybe_unused]] double x, [[maybe_unused]] double y, [[maybe_unused]] double z, [[maybe_unused]] double gamma){
        return std::sqrt(x*x+y*y+z*z) / gamma;
    };

    expect(g.CalculateMoment(w,v) < 0.0001f) << "Expected < 0.0001, got " << g.CalculateMoment(w,v);
    expect(g.CalculateMoment(w,v) >= 0.00005f) << "Expected > 0.00005, got " << g.CalculateMoment(w,v);
  };
};

}  // namespace

int
  main(int argc, const char** argv)
{
  return static_cast<int>(cfg<override>.run(run_cfg { .argc = argc, .argv = argv }));
}
