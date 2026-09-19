## Chris Wilhelm - CS5722 - Homework 1 Reflection - 2026-09-19

---

Why does vectorization beat explicit Python loops?
Address what happens under the hood in NumPy versus a for loop over a Python list, and give one concrete example from this assignment where boradcasting replaced a nested loop.

---

Most of the discussion in session three entertains this.  Vectorization beats explicit loops because of the difference in overhead and compiled code math execution.  Rather than allocating heap space and pointers, checking data types, and then operating on the values one at a time iterating through a loop, numpy will perform the math on whole arrays at once.  Also, as noted in session 3 "When NumPy executes a * b , the dtype and broadcasting work is resolved for whole arrays, and a compiled loop traverses their strided buffers as fast as the layout permits. Contiguous layouts are especially friendly to caches and SIMD; non-contiguous views may be slower but still avoid a Python dispatch for every element."
One example in the homework where broadcasting replaced a nested loop is in the pairwise distances method.  Instead of looping through each element of each array, calculating the differences, we use a method call that subtracts all elements of both arrays, simultaneously.
```
for i in range(len(A)):
    for j in range(len(B)):
        difference = A[i] - B[j]
```
turns into
```
    differenceofelements = A[:, None, :] - B[None, :, :] 
```