SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I../nunn

hopfield_test_CXXFLAGS = \
   -std=c++11 \
   -g 

hopfield_test_SOURCES = \
   hopfield_test.cc

AM_CXXFLAGS = ${hopfield_test_CXXFLAGS}

hopfield_test_LDADD = \
   -lnunn 

sbin_PROGRAMS += \
   hopfield_test 


