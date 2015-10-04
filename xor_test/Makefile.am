SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I../nunn

xor_test_CXXFLAGS = \
   -std=c++11 \
   -g 

xor_test_SOURCES = \
   xor_test.cc

AM_CXXFLAGS = ${xor_test_CXXFLAGS}

xor_test_LDADD = \
   -lnunn 

sbin_PROGRAMS += \
   xor_test 


