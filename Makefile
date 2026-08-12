CC := clang
CPPFLAGS :=
CFLAGS := -O3 -funroll-loops -march=native
LDLIBS := -lopenblas -lm

ifeq ($(DEBUG),1)
CPPFLAGS += -DDEBUG
endif

BUILD_DIR := .build
VARIANT_SRCS := $(wildcard variants/0[0-4]_*.c)
BASE_SRCS := gemm.c $(VARIANT_SRCS)
BASE_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(BASE_SRCS))
PLOT_OBJS := $(BUILD_DIR)/plot.o $(patsubst %.c,$(BUILD_DIR)/%.o,$(VARIANT_SRCS))
AVX512_OBJ := $(BUILD_DIR)/variants/05_outer_product_avx512.o
OBJS := $(BASE_OBJS) $(AVX512_OBJ)

.PHONY: all clean

all: gemm

gemm: $(OBJS)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

plot: $(PLOT_OBJS) $(AVX512_OBJ)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(AVX512_OBJ): CFLAGS += -mavx512f

clean:
	rm -rf $(BUILD_DIR) gemm plot
