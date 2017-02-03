SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I$(top_srcdir)/nunn

maze_CXXFLAGS = \
   -std=c++11 \
   -g 

maze_SOURCES = \
   maze.cc

AM_CXXFLAGS = ${maze_CXXFLAGS}

maze_LDADD = \
   -L../../nunn/.libs -lnunn 

sbin_PROGRAMS += \
   maze 


