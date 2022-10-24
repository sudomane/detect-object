#ifndef ERROR_HPP
#define ERROR_HPP

void _abortError(const char* msg, const char* fname, int line);

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

#endif // ERR_HPP