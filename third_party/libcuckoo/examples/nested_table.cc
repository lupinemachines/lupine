/* We demonstrate how to nest hash tables within one another, to store
 * unstructured data, kind of like JSON. There's still the limitation that it's
 * statically typed. */

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <libcuckoo/cuckoohash_map.hh>

typedef libcuckoo::cuckoohash_map<std::string, std::string> InnerTable;
typedef libcuckoo::cuckoohash_map<std::string, std::unique_ptr<InnerTable>> OuterTable;

int main() {
  OuterTable tbl;

  tbl.insert("bob", std::unique_ptr<InnerTable>(new InnerTable));
  tbl.update_fn("bob", [](std::unique_ptr<InnerTable> &innerTbl) {
    innerTbl->insert("nickname", "jimmy");
    innerTbl->insert("pet", "dog");
    innerTbl->insert("food", "bagels");
  });

  tbl.insert("jack", std::unique_ptr<InnerTable>(new InnerTable));
  tbl.update_fn("jack", [](std::unique_ptr<InnerTable> &innerTbl) {
    innerTbl->insert("friend", "bob");
    innerTbl->insert("activity", "sleeping");
    innerTbl->insert("language", "javascript");
  });

  {
    auto lt = tbl.lock_table();
    for (const auto &item : lt) {
      std::cout << "Properties for " << item.first << std::endl;
      auto innerLt = item.second->lock_table();
      for (auto innerItem : innerLt) {
        std::cout << "\t" << innerItem.first << " = " << innerItem.second
                  << std::endl;
      }
    }
  }
}
