SUBDIRS =
sbin_PROGRAMS = 

AM_CPPFLAGS = \
   -g \
   -I../nunn

tictactoe_CXXFLAGS = \
   -std=c++11 \
   -g 

tictactoe_SOURCES = \
   tictactoe.cc

AM_CXXFLAGS = ${tictactoe_CXXFLAGS}

tictactoe_LDADD = \
   -lnunn 

sbin_PROGRAMS += \
   tictactoe 


