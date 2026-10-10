int leaf_value = 40;
__attribute__((constructor)) static void init(void) { leaf_value = 41; }
int leaf_add(int value) { return leaf_value + value; }
