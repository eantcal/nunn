SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I$(top_srcdir)/mnist \
   -I$(top_srcdir)/nunn

mnist_test_CXXFLAGS = \
   -std=c++11 \
   -g 

mnist_test_SOURCES = \
   mnist_test.cc

AM_CXXFLAGS = ${mnist_test_CXXFLAGS}

mnist_test_LDADD = \
   -L../../nunn/.libs -lnunn \
   -L../../mnist/.libs -lmnist

sbin_PROGRAMS += \
   mnist_test 


