// This file includes the acutest runner without the TEST_NO_MAIN macro.
#include "acutest.h"

#include "test_bucket_container.h"
#include "test_c_interface.h"
#include "test_constructor.h"
#include "test_hash_properties.h"
#include "test_heterogeneous_compare.h"
#include "test_iterator.h"
#include "test_locked_table.h"
#include "test_maximum_hashpower.h"
#include "test_minimum_load_factor.h"
#include "test_noncopyable_types.h"
#include "test_resize.h"
#include "test_user_exceptions.h"

TEST_LIST = {

    {"test_bucket_container_default_constructor",
     test_bucket_container_default_constructor},
    {"test_bucket_container_simple_stateful_allocator",
     test_bucket_container_simple_stateful_allocator},
    {"test_bucket_container_copy_construction",
     test_bucket_container_copy_construction},
    {"test_bucket_container_move_construction",
     test_bucket_container_move_construction},
    {"test_bucket_container_copy_assignment_with_propagate",
     test_bucket_container_copy_assignment_with_propagate},
    {"test_bucket_container_copy_assignment_no_propagate",
     test_bucket_container_copy_assignment_no_propagate},
    {"test_bucket_container_move_assignment_with_propagate",
     test_bucket_container_move_assignment_with_propagate},
    {"test_bucket_container_move_assignment_no_propagate_equal",
     test_bucket_container_move_assignment_no_propagate_equal},
    {"test_bucket_container_move_assignment_no_propagate_unequal",
     test_bucket_container_move_assignment_no_propagate_unequal},
    {"test_bucket_container_swap_no_propagate",
     test_bucket_container_swap_no_propagate},
    {"test_bucket_container_swap_propagate",
     test_bucket_container_swap_propagate},
    {"test_bucket_container_setKV_with_throwing_type_maintains_strong_"
     "guarantee",
     test_bucket_container_setKV_with_throwing_type_maintains_strong_guarantee},
    {"test_bucket_container_copy_assignment_with_throwing_type_is_destroyed_"
     "properly",
     test_bucket_container_copy_assignment_with_throwing_type_is_destroyed_properly},
    {"test_bucket_container_copy_destroyed_buckets_container",
     test_bucket_container_copy_destroyed_buckets_container},

    {"test_c_interface_basic", test_c_interface_basic},
    {"test_c_interface_locked_table", test_c_interface_locked_table},

    {"test_constructor_default_size", test_constructor_default_size},
    {"test_constructor_given_size", test_constructor_given_size},
    {"test_constructor_frees_even_with_exceptions",
     test_constructor_frees_even_with_exceptions},
    {"test_constructor_stateful_components",
     test_constructor_stateful_components},
    {"test_constructor_range_constructor", test_constructor_range_constructor},
    {"test_constructor_copy_constructor", test_constructor_copy_constructor},
    {"test_constructor_copy_constructor_other_allocator",
     test_constructor_copy_constructor_other_allocator},
    {"test_constructor_move_constructor", test_constructor_move_constructor},
    {"test_constructor_move_constructor_different_allocator",
     test_constructor_move_constructor_different_allocator},
    {"test_constructor_initializer_list_constructor",
     test_constructor_initializer_list_constructor},
    {"test_constructor_swap_maps", test_constructor_swap_maps},
    {"test_constructor_copy_assign_different_allocators",
     test_constructor_copy_assign_different_allocators},
    {"test_constructor_move_assign_different_allocators",
     test_constructor_move_assign_different_allocators},
    {"test_constructor_move_assign_same_allocators",
     test_constructor_move_assign_same_allocators},
    {"test_constructor_initializer_list_assignment",
     test_constructor_initializer_list_assignment},

    {"test_hash_properties_int_alt_index_works_correctly",
     test_hash_properties_int_alt_index_works_correctly},
    {"test_hash_properties_string_alt_index_works_correctly",
     test_hash_properties_string_alt_index_works_correctly},
    {"test_hash_properties_hash_with_larger_hashpower_only_adds_top_bits",
     test_hash_properties_hash_with_larger_hashpower_only_adds_top_bits},

    {"test_heterogeneous_compare", test_heterogeneous_compare},

    {"test_iterator_types", test_iterator_types},
    {"test_iterator_empty_table_iteration",
     test_iterator_empty_table_iteration},
    {"test_iterator_walkthrough", test_iterator_walkthrough},
    {"test_iterator_modification", test_iterator_modification},
    {"test_iterator_lock_table_blocks_inserts",
     test_iterator_lock_table_blocks_inserts},
    {"test_iterator_cast_iterator_to_const_iterator",
     test_iterator_cast_iterator_to_const_iterator},

    {"test_locked_table_typedefs", test_locked_table_typedefs},
    {"test_locked_table_move", test_locked_table_move},
    {"test_locked_table_unlock", test_locked_table_unlock},
    {"test_locked_table_info", test_locked_table_info},
    {"test_locked_table_clear", test_locked_table_clear},
    {"test_locked_table_insert_duplicate", test_locked_table_insert_duplicate},
    {"test_locked_table_insert_new_key", test_locked_table_insert_new_key},
    {"test_locked_table_insert_lifetime", test_locked_table_insert_lifetime},
    {"test_locked_table_erase", test_locked_table_erase},
    {"test_locked_table_find", test_locked_table_find},
    {"test_locked_table_at", test_locked_table_at},
    {"test_locked_table_operator_brackets",
     test_locked_table_operator_brackets},
    {"test_locked_table_count", test_locked_table_count},
    {"test_locked_table_equal_range", test_locked_table_equal_range},
    {"test_locked_table_rehash", test_locked_table_rehash},
    {"test_locked_table_reserve", test_locked_table_reserve},
    {"test_locked_table_equality", test_locked_table_equality},
    {"test_locked_table_holds_locks_after_resize",
     test_locked_table_holds_locks_after_resize},
    {"test_locked_table_io", test_locked_table_io},
    {"test_empty_locked_table_io", test_empty_locked_table_io},

    {"test_maximum_hashpower_initialized_to_default",
     test_maximum_hashpower_initialized_to_default},
    {"test_maximum_hashpower_caps_any_expansion",
     test_maximum_hashpower_caps_any_expansion},
    {"test_maximum_hash_power_none", test_maximum_hash_power_none},

    {"test_minimum_load_factor_initialized_to_default",
     test_minimum_load_factor_initialized_to_default},
    {"test_minimum_load_factor_caps_automatic_expansion",
     test_minimum_load_factor_caps_automatic_expansion},
    {"test_minimum_load_factor_invalid", test_minimum_load_factor_invalid},

    {"test_noncopyable_types_insert_and_update",
     test_noncopyable_types_insert_and_update},
    {"test_noncopyable_types_insert_or_assign",
     test_noncopyable_types_insert_or_assign},
    {"test_noncopyable_types_upsert", test_noncopyable_types_upsert},
    {"test_noncopyable_types_iteration", test_noncopyable_types_iteration},
    {"test_noncopyable_types_nested_table",
     test_noncopyable_types_nested_table},
    {"test_noncopyable_types_insert_lifetime",
     test_noncopyable_types_insert_lifetime},
    {"test_noncopyable_types_erase_fn", test_noncopyable_types_erase_fn},
    {"test_noncopyable_types_uprase_fn", test_noncopyable_types_uprase_fn},

    {"test_resize_rehash_empty_table", test_resize_rehash_empty_table},
    {"test_resize_reserve_empty_table", test_resize_reserve_empty_table},
    {"test_resize_reserve_calc", test_resize_reserve_calc},
    {"test_resize_number_of_frees", test_resize_number_of_frees},
    {"test_resize_on_non_relocatable_type",
     test_resize_on_non_relocatable_type},

    {"test_user_exceptions", test_user_exceptions},

    {NULL, NULL},
};
