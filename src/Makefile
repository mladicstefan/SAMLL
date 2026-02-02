CC = gcc
CFLAGS = -g -Wall -lrt -Wextra -Wpedantic -std=c11 \
         -fsanitize=address,undefined \
         -fno-omit-frame-pointer \
         -DDEBUG
LDFLAGS = -lm -fsanitize=address,undefined

LIB_CFLAGS = -fPIC -O2 -I./include
LIB_DIR = ../shared
LIB_TARGET = $(LIB_DIR)/libmatrix.so

SRC = matrix.c
TARGET = matrix

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -I./include -o $(TARGET) $(SRC) $(LDFLAGS)

lib: $(LIB_TARGET)

$(LIB_TARGET): $(SRC) | $(LIB_DIR)
	$(CC) -shared $(LIB_CFLAGS) -o $(LIB_TARGET) $(SRC)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

clean:
	rm -f $(TARGET) $(LIB_TARGET)
