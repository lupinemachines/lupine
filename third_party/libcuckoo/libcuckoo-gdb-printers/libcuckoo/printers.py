import gdb
import gdb.printing

class CuckoohashMapPrinter:
 """Print a cuckoohash_map object"""

 def __init__(self, val):
   self.val = val
   self.slot_per_bucket = int(self.val.type.template_argument(5))

 def _iterator(self):
   buckets_obj = self.val['buckets_']
   hashpower = buckets_obj['hashpower_']['_M_i']
   buckets_ptr = buckets_obj['buckets_']
   if not buckets_ptr:
     return
   num_buckets = int(2**hashpower)
   for i in range(num_buckets):
     bucket = (buckets_ptr + i).dereference()
     storage_value_type = gdb.lookup_type(str(bucket.type) + '::storage_value_type')
     for j in range(self.slot_per_bucket):
       if bucket['occupied_']['_M_elems'][j]:
         value_blob = bucket['values_']['_M_elems'][j]
         storage_value = value_blob.cast(storage_value_type)
         yield ('key', storage_value['first'])
         yield ('value', storage_value['second'])

 def children(self):
   return self._iterator()

 def to_string(self):
   return 'cuckoohash_map'

 def display_hint(self):
   return 'map'

def build_pretty_printer():
  pp = gdb.printing.RegexpCollectionPrettyPrinter("libcuckoo")
  pp.add_printer('cuckoohash_map', '^cuckoohash_map<.*>$', CuckoohashMapPrinter)
  return pp

def register_libcuckoo_printers(objfile):
  gdb.printing.register_pretty_printer(objfile, build_pretty_printer())
