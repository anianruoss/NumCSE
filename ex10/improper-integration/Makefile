PROJNAME := improper-integration 

CPP := g++
CFLAGS := -Wall -Wextra -Wpedantic -std=gnu++14 -I/usr/include/eigen3
LFLAGS := -lstdc++ -lmgl

ifeq (1,${DEBUG})
	CFLAGS += -O0 -g
	OUTDIR := bin/out/debug
else
	CFLAGS += -O3 -march=native 
	OUTDIR := bin/out/release
endif

CPPFILES := $(wildcard *.cpp)
OBJFILES := $(addprefix bin/obj/, $(CPPFILES:%.cpp=%.o))

$(info Objects: ${OBJFILES})

.PHONY := clean dirs

all: dirs proj

proj: ${OBJFILES}
	@echo "Linking: "${PROJNAME}
	${CPP} -o ${OUTDIR}/${PROJNAME} $? ${LFLAGS}

bin/obj/%.o: %.cpp
	@echo "Compiling "$<
	${CPP} -o $@ -c ${CFLAGS} $<

clean: 
	@rm -r ./bin 2>/dev/null || true
	@rm ${OBJFILES} 2>/dev/null || true
	@rm ${PROJNAME} 2>/dev/null || true
	@rm -r plots 2>/dev/null || true

dirs:
	mkdir -p bin/obj ${OUTDIR}
	mkdir -p plots

