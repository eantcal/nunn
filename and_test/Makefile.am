SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I../nunn

and_test_CXXFLAGS = \
   -std=c++11 \
   -g 

and_test_SOURCES = \
   and_test.cc

AM_CXXFLAGS = ${and_test_CXXFLAGS}

and_test_LDADD = \
   -lnunn 

sbin_PROGRAMS += \
   and_test 


