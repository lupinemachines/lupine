#include "visible_devices.h"

#include <cassert>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Four devices whose UUIDs differ only in the last byte.
std::string fake_uuid(int index) {
  std::string hex(31, 'a');
  hex.push_back(static_cast<char>('0' + index));
  return hex;
}

std::vector<int> select(const char *spec, int count = 4) {
  return lupine_select_visible_devices(spec, count, fake_uuid);
}

void expect(const std::vector<int> &got, const std::vector<int> &want,
            const char *what) {
  if (got != want) {
    std::string g, w;
    for (int v : got) {
      g += std::to_string(v) + " ";
    }
    for (int v : want) {
      w += std::to_string(v) + " ";
    }
    fprintf(stderr, "%s: got [%s], want [%s]\n", what, g.c_str(), w.c_str());
    assert(false);
  }
}

void test_unset_shows_everything() {
  expect(select(nullptr), {0, 1, 2, 3}, "unset");
}

void test_empty_and_minus_one_show_nothing() {
  expect(select(""), {}, "empty");
  expect(select("-1"), {}, "-1");
}

void test_indices_select_and_reorder() {
  expect(select("0,2"), {0, 2}, "subset");
  // The order given is the order the client sees, so a device can be moved to
  // ordinal 0 without touching the fleet.
  expect(select("3,1"), {3, 1}, "reordering");
  expect(select("1"), {1}, "single");
}

void test_entries_naming_nothing_are_dropped() {
  expect(select("0,9"), {0}, "out of range");
  expect(select("0,,2"), {0, 2}, "empty entry");
  expect(select("0, 2"), {0, 2}, "spaces");
  expect(select("abc"), {}, "not an index or a uuid");
}

void test_uuids_match_however_they_are_written() {
  const std::string bare = fake_uuid(2);
  expect(select(bare.c_str()), {2}, "bare uuid");
  expect(select(("GPU-" + bare).c_str()), {2}, "prefixed uuid");

  // Hyphenated in the usual 8-4-4-4-12 shape, and upper case.
  std::string hyphenated = "GPU-";
  for (size_t i = 0; i < bare.size(); ++i) {
    if (i == 8 || i == 12 || i == 16 || i == 20) {
      hyphenated.push_back('-');
    }
    hyphenated.push_back(static_cast<char>(toupper(bare[i])));
  }
  expect(select(hyphenated.c_str()), {2}, "hyphenated upper case uuid");

  expect(select((fake_uuid(3) + "," + fake_uuid(0)).c_str()), {3, 0},
         "uuids keep the order given");
  std::string unknown(32, 'f');
  expect(select(unknown.c_str()), {}, "uuid matching no device");
}

void test_indices_and_uuids_mix() {
  expect(select(("1," + fake_uuid(3)).c_str()), {1, 3}, "mixed");
}

// A device whose UUID cannot be read matches no UUID, but is still reachable
// by index.
void test_unreadable_uuid() {
  auto silent = [](int) { return std::string(); };
  expect(lupine_select_visible_devices(fake_uuid(1).c_str(), 4, silent), {},
         "unreadable uuid matches nothing");
  expect(lupine_select_visible_devices("1", 4, silent), {1},
         "index still works");
}

void test_no_devices_to_choose_from() {
  expect(select(nullptr, 0), {}, "unset with no devices");
  expect(select("0", 0), {}, "index with no devices");
}

} // namespace

int main() {
  test_unset_shows_everything();
  test_empty_and_minus_one_show_nothing();
  test_indices_select_and_reorder();
  test_entries_naming_nothing_are_dropped();
  test_uuids_match_however_they_are_written();
  test_indices_and_uuids_mix();
  test_unreadable_uuid();
  test_no_devices_to_choose_from();
  printf("visible_devices_test passed\n");
  return 0;
}
