SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I../mnist \
   -I../nunn

# IRSC Bus Topology Basic Unit Test

mnist_test_CXXFLAGS = \
   -std=c++11 \
   -g 

mnist_test_SOURCES = \
   mnist_test.cc

AM_CXXFLAGS = ${mnist_test_CXXFLAGS}

mnist_test_LDADD = \
   -lnunn \
   -lmnist

sbin_PROGRAMS += \
   mnist_test 


