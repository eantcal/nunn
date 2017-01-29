SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I$(top_srcdir)/nunn

path_finder_CXXFLAGS = \
   -std=c++11 \
   -g 

path_finder_SOURCES = \
   path_finder.cc

AM_CXXFLAGS = ${path_finder_CXXFLAGS}

path_finder_LDADD = \
   -L../../nunn/.libs -lnunn 

sbin_PROGRAMS += \
   path_finder 


