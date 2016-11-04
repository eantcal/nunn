SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I$(top_srcdir)/nunn

counter_test_CXXFLAGS = \
   -std=c++11 \
   -g 

counter_test_SOURCES = \
   counter_test.cc

AM_CXXFLAGS = ${counter_test_CXXFLAGS}

counter_test_LDADD = \
   -L../nunn/.libs -lnunn 

sbin_PROGRAMS += \
   counter_test 


