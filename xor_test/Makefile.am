SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I$(top_srcdir)/nunn

xor_test_CXXFLAGS = \
   -std=c++11 \
   -g 

xor_test_SOURCES = \
   xor_test.cc

AM_CXXFLAGS = ${xor_test_CXXFLAGS}

xor_test_LDADD = \
   -L../nunn/.libs -lnunn 

sbin_PROGRAMS += \
   xor_test 


