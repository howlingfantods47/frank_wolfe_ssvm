DIRS := .

SOURCES := $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cpp))
SOURCES_MAIN := $(filter-out %/OCR.cpp %/USPS.cpp %/LSIS.cpp, $(SOURCES))
OBJS := $(patsubst %.cpp, %.o, $(SOURCES))
OBJS_MAIN := $(patsubst %.cpp, %.o, $(SOURCES_MAIN))
USPS_SRC := ./USPS.cpp $(SOURCES_MAIN)
OCR_SRC := ./OCR.CPP $(SOURCES_MAIN)
LSIS_SRC := ./LSIS.cpp $(SOURCES_MAIN)
#$(info $(USPS_SRC))
USPS_OBJS := $(patsubst %.cpp, %.o, $(USPS_SRC))
OCR_OBJS := $(patsubst %.cpp, %.o, $(OCR_SRC))
LSIS_OBJS := $(patsubst %.cpp, %.o, $(LSIS_SRC))


CFLAGS := -O0 -ggdb
#CFLAGS := -O3 -D_NDEBUG 
CXX ?= g++
LIBS := 
INCLUDES := 
LIBDIR := 

# Add librt if the target platform is not Darwin (OS X)
ifneq ($(shell uname -s),Darwin)
	LIBS += -lrt
endif

#all: svm_train

svm_train: ${OBJS}
	$(CXX) $(CFLAGS) ${LIBDIR} -o $@ ${OBJS} ${LIBS}

usps: ${USPS_OBJS}
	$(CXX) $(CFLAGS) ${LIBDIR} -o $@ ${USPS_OBJS} ${LIBS}

ocr: ${OCR_OBJS}
	$(CXX) $(CFLAGS) ${LIBDIR} -o $@ ${OCR_OBJS} ${LIBS}

lsis: ${LSIS_OBJS}
	$(CXX) $(CFLAGS) ${LIBDIR} -o $@ ${LSIS_OBJS} ${LIBS}

.cpp.o:
	$(CXX) $(CFLAGS) ${INCLUDES} $< -c -o $@

clean:
	rm -f ${OBJS} svm_train ocr usps lsis
