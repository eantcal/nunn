SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I$(top_srcdir)/nunn

nunn_topo_CXXFLAGS = \
   -std=c++14 \
   -g 

nunn_topo_SOURCES = \
   nunn_topo.cc

AM_CXXFLAGS = ${nunn_topo_CXXFLAGS}

nunn_topo_LDADD = \
   -L../nunn/.libs -lnunn 

sbin_PROGRAMS += \
   nunn_topo 


