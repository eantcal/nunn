SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I../nunn

xorfunc_CXXFLAGS = \
   -std=c++11 \
   -g 

xorfunc_SOURCES = \
   xorfunc.cc

AM_CXXFLAGS = ${xorfunc_CXXFLAGS}

xorfunc_LDADD = \
   -lnunn 

sbin_PROGRAMS += \
   xorfunc 


