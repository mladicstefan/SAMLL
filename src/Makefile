CC = gcc
CFLAGS_COMMON = -Wall -Wextra -Wpedantic -std=c11 -I./include
CFLAGS_DEBUG = $(CFLAGS_COMMON) -g -fsanitize=address,undefined \
               -fno-omit-frame-pointer -DDEBUG
CFLAGS_RELEASE = $(CFLAGS_COMMON) -O3 -DNDEBUG -march=native
LDFLAGS_DEBUG = -fsanitize=address,undefined -lm -lrt
LDFLAGS_RELEASE = -lm -lrt

SRCS = nn.c dataset.c matrix.c include/random.c
OBJS_DEBUG = $(SRCS:.c=.debug.o)
OBJS_RELEASE = $(SRCS:.c=.release.o)
TARGET_DEBUG = nn_debug
TARGET_RELEASE = nn

all: debug

debug: CFLAGS = $(CFLAGS_DEBUG)
debug: LDFLAGS = $(LDFLAGS_DEBUG)
debug: $(TARGET_DEBUG)

$(TARGET_DEBUG): $(OBJS_DEBUG)
	$(CC) $(OBJS_DEBUG) -o $(TARGET_DEBUG) $(LDFLAGS_DEBUG)

%.debug.o: %.c
	$(CC) $(CFLAGS_DEBUG) -c $< -o $@

release: CFLAGS = $(CFLAGS_RELEASE)
release: LDFLAGS = $(LDFLAGS_RELEASE)
release: $(TARGET_RELEASE)

$(TARGET_RELEASE): $(OBJS_RELEASE)
	$(CC) $(OBJS_RELEASE) -o $(TARGET_RELEASE) $(LDFLAGS_RELEASE)

%.release.o: %.c
	$(CC) $(CFLAGS_RELEASE) -c $< -o $@

clean:
	rm -f $(TARGET_DEBUG) $(TARGET_RELEASE) $(OBJS_DEBUG) $(OBJS_RELEASE)

.PHONY: all debug release clean
