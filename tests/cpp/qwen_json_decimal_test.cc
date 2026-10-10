#include <atomic>
#include <cassert>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "../../cpp/support/json_parse.h"

static picojson::value Parse(const std::string& text) {
  picojson::value value;
  assert(xgrammar::ParseJSON(value, text).empty());
  return value;
}

int main() {
  const std::string decimal = "0.10000000000000000001";
  auto original = Parse(decimal);
  assert(original.number_lexeme() == decimal);
  assert(original.serialize() == decimal);
  auto copied = original;
  auto moved = std::move(copied);
  assert(moved.serialize() == decimal);
  picojson::value assigned;
  assigned = original;
  assert(assigned.serialize() == decimal);
  picojson::value move_assigned;
  move_assigned = std::move(assigned);
  assert(move_assigned.serialize() == decimal);
  auto swapped = Parse("-1.234567890123456789e-30");
  moved.swap(swapped);
  assert(swapped.serialize() == decimal);
  assert(moved.serialize() == "-1.234567890123456789e-30");
  auto large = Parse("1000000000000000000000000000001");
  assert(large.serialize() == "1000000000000000000000000000001");
  for (const auto& text : {"1e100000", "1e-2147483648", "0e999999999999999999999"}) {
    auto value = Parse(text);
    assert(value.serialize() == text);
  }
  for (const auto& text : {"01", "1.", "-01", "1e", "1e+", "--1", "1+2"}) {
    picojson::value invalid;
    assert(!xgrammar::ParseJSON(invalid, std::string(text)).empty());
  }
  auto nested = Parse("{\"x\":[0.10000000000000000001,1000000000000000000000000000001]}");
  auto nested_copy = nested;
  auto& array = nested_copy.get<picojson::object>().at("x").get<picojson::array>();
  for (int i = 0; i < 1024; ++i) array.push_back(original);
  const auto& first = array.front();
  assert(first.number_lexeme() == decimal);
  assert(first.serialize() == decimal);
  assert(nested.serialize() == "{\"x\":[0.10000000000000000001,1000000000000000000000000000001]}");
  auto mutated = original;
  mutated.get<double>() = 0.5;
  assert(mutated.number_lexeme().empty());
  assert(mutated.serialize() == "0.5");
  mutated = original;
  const double quarter = 0.25;
  mutated.set<double>(quarter);
  assert(mutated.number_lexeme().empty());
  assert(mutated.serialize() == "0.25");
  // Both production guarded parsing and picojson's own contexts preserve values.
  picojson::value upstream;
  assert(picojson::parse(upstream, decimal).empty());
  assert(upstream.serialize() == decimal);
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < 1000; ++i) {
        auto parsed = Parse(decimal);
        auto copy = parsed;
        if (copy.serialize() != decimal || original.serialize() != decimal) ++failures;
      }
    });
  }
  for (auto& thread : threads) thread.join();
  assert(failures == 0);
  std::cout << "production ParseJSON decimal/copy/move/swap/array/serialize/mutation/8-thread checks passed\n";
}
