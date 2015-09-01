#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([nunn], [1.47], [acaldmail@gmail.com])
AM_INIT_AUTOMAKE
AC_CONFIG_HEADERS([config.h])

AC_CONFIG_MACRO_DIR([m4])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LIBTOOL
LT_INIT



# Checks for header files.
AC_CHECK_HEADERS([malloc.h memory.h stdlib.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_INT32_T
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_CHECK_FUNCS([sqrt])

AC_CONFIG_FILES([
  Makefile
  nunn/Makefile
  mnist/Makefile
  mnist_test/Makefile
  tictactoe/Makefile
])
AC_OUTPUT
