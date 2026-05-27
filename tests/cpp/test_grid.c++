// Copyright 2026 - 2026, Miro Palmu, Joonas Nättilä and the runko contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <boost/ut.hpp>
#include "runko/mdgrid_common.h"
#include "runko/vlv/vlasov_grid.h"

namespace {

using namespace boost::ut;
[[maybe_unused]]
const suite<"grid testing"> _ = [] {
  "grid"_test = [] {
    runko::ScalarGrid<float> grid = runko::ScalarGrid<float>(3,4,5);

    expect(true);
  };

  "vlasov_grid"_test = [] {
    auto _ = vlv::DenseGrid(5,4,5);
    expect(true); 
  };
};

}  // namespace

int
  main(int argc, const char** argv)
{
  return static_cast<int>(cfg<override>.run(run_cfg { .argc = argc, .argv = argv }));
}
