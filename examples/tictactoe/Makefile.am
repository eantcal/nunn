SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I$(top_srcdir)/nunn

tictactoe_CXXFLAGS = \
   -std=c++11 \
   -g 

tictactoe_SOURCES = \
   tictactoe.cc

AM_CXXFLAGS = ${tictactoe_CXXFLAGS}

tictactoe_LDADD = \
   -L../../nunn/.libs -lnunn 

sbin_PROGRAMS += \
   tictactoe 


