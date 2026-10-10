extern int leaf_add(int);
int (*dependency_callback)(int) = leaf_add;
int middle_add(int value) { return dependency_callback(value) + 1; }
