---
layout: post
title:  Why is Numpy Faster than Pure Python?
date: 2024-02-02 08:20:00
description: TLDR; Numpy leverages contiguous memory allocation and vectorizes operations over entire arrays.
tags: 
categories: 
---

**TLDR**; Numpy leverages contiguous memory allocation and vectorizes operations over entire arrays.


I've used Numpy since my early programming days in Python during my undergrad in physics. I was taught we should always use Numpy when dealing with numeric computations like matrix multiplication. I knew Numpy has optimized C implementations of some operations, but I wanted to dig deeper into what's going on under the hood to make Numpy more efficient than pure Python for matrix-like structures. After all, Python also runs C in the background. 

I found good answers in the [High Performance Python book](https://www.oreilly.com/library/view/high-performance-python/9781492055013/), which I decided to summarize in this post.

## Vectorization

Let’s assume for a moment that we can have all the data we need to run an operation on the CPU. Vectorization is a process where you apply the same operation simultaneously to multiple elements. For example, you can multiply parts of arrays at once instead of multiplying element by element. 

The caveat of vectorized operations is that they run on a different part of the CPU and with different instructions than non-vectorized operations. Python doesn't leverage vectorized operations that are possible in the CPU, so Numpy has specialized code that takes advantage of the vectorizations enabled by the CPU. That’s why vectorization is one of the reasons Numpy is faster than pure Python.

## Fragmented vs. Contiguous Memory

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/numpy_vs_python.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fragmented memory results in irrelevant data (green squares) in each memory transfer, so we need more memory transfers to pass the relevant data (blue squares) to the CPU cache.
</div>


Fragmented memory results in irrelevant data (green squares) in each memory transfer, so we need more memory transfers to pass the relevant data (blue squares) to the CPU cache.

We now know that vectorization requires all the data in the CPU. However, CPUs have limited memory, so we need to figure out how to transfer data between the RAM and the CPU’s cache.

Python lists and Numpy arrays handle data differently. Python lists store pointers to the data; meaning lists don't hold the data we care about. Storing pointers of the data allows Python to hold multiple types of data in a list, resulting in our relevant data being fragmented in different memory locations. This fragmentation is fine for most cases, but a potential optimization becomes clear when we understand communications between the RAM and the CPU.

The data we use is initially stored in RAM and then moved to the CPU when we need to run calculations with that data. Communication between the two devices is costly in terms of time, so CPUs have a cache memory where they can store data they know they will require to run the calculations we request. That's why the CPU tries to predict which data will be required and tries to transfer that data to its cache. The CPU usually makes this prediction well (using techniques like branch prediction and pipelining), so the real bottleneck is moving data quickly between the RAM and the CPU cache.

Transferring between the RAM and the CPU cache—also known as the L1/L2 cache—is done by a bus that transfers memory in blocks. We usually want to run an operation over the entire object for data structures like matrices, so we know that, eventually, we want to transfer all the elements in the matrix to the CPU. If the data we want to use is fragmented across our RAM, the transferred blocks will contain pieces that are not relevant to our calculation. If our data is stored in contiguous blocks, most of our data will be relevant to the calculation. That's why we need fewer transfer operations than in the fragmented memory case. Using contiguous memory gets our relevant data to the CPU faster, and this optimization results in faster runtimes. You can see a comparison of the two cases in the diagram below.

## Takeaways

Python is a very flexible language, but that flexibility often comes with a price we pay in performance. In our case, Python allows having lists containing elements of different types, which are challenging to store in contiguous memory without causing issues. Numpy then enforces elements in an array to be the same type to speed up calculations by reducing data transfers and leveraging vectorization. That limitation of having arrays of the same data type can be limiting for some use cases but isn’t a problem for numeric computing. That’s why Numpy works great for handling operations between numerical tensors and is faster than pure Python for those cases.