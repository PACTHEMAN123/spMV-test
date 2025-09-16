# cuda kernel for sparse SGEMV

## build

to build this project

```bash
cmake -S . -B build/
cd build && make
```

## test

to run the test

```bash
cd build && ./sparse_sgemv
```

to get the profile of sgemv

```bash
chmod +x profile.sh
./profile.sh
```

### kernels

notice that the operation is:

$$
Y = XA
$$

where Y donates for N, X donates for M, A donates for (M, N)

