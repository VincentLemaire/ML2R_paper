#-------------------------------------------------
#
# Project created by QtCreator 2014-04-10T17:32:11
#
#-------------------------------------------------

QT       -= core gui

TARGET = np11
TEMPLATE = lib

DEFINES += NP11_LIBRARY

SOURCES += \
    fcts.cpp

HEADERS +=\
    distributions.hpp \
    fcts.hpp \
    linear_estimator.hpp \
    modeles.hpp \
    scheme.hpp \
    time.hpp \
    structural_parameters.hpp \
    multilevel_parameters.hpp \
    multilevel_estimators.hpp \
    stochastic_algorithm.hpp \
    scheme2.hpp

unix {
    target.path = /usr/lib
    INSTALLS += target
}
