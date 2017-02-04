sbin_PROGRAMS = maze mazesarsa

AM_CPPFLAGS = \
   -g \
   -I$(top_srcdir)/nunn

maze_CXXFLAGS = \
   -std=c++11 \
   -g 

maze_SOURCES = \
   maze.cc

maze_LDADD = \
   -L../../nunn/.libs -lnunn 

mazesarsa_CXXFLAGS = \
   -std=c++11 -DUSE_SARSA \
   -g 

mazesarsa_SOURCES = \
   maze.cc

mazesarsa_LDADD = \
   -L../../nunn/.libs -lnunn 



