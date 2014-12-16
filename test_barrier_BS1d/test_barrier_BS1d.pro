TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CXXFLAGS += -pthread -std=c++11
QMAKE_LFLAGS += -Wl,--no-as-needed
LIBS += -pthread
# LIBS += -llapack -lblas -larmadillo

QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3 #-ffast-math

QMAKE_LFLAGS_RELEASE -= -O1
QMAKE_LFLAGS_RELEASE += -O3 #-ffast-math

SOURCES += \
    main.cpp


win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../np11/release/ -lnp11
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../np11/debug/ -lnp11
else:unix: LIBS += -L$$OUT_PWD/../np11/ -lnp11

INCLUDEPATH += $$PWD/../np11
DEPENDPATH += $$PWD/../np11

