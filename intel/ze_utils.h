int read_from_binary(unsigned char **output, size_t *size, const char *name);
void check_error(ze_result_t error, char const *name);
void exit_msg(char const *str);