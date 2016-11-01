SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I../nunn

counter_test_CXXFLAGS = \
   -std=c++11 \
   -g 

counter_test_SOURCES = \
   counter_test.cc

AM_CXXFLAGS = ${counter_test_CXXFLAGS}

counter_test_LDADD = \
   -lnunn 

sbin_PROGRAMS += \
   counter_test 


