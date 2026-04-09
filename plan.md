# Plan

Our goal is optimize the bottlenecks of games for which there are open source PC ports. As a beginning step, we will look at the `tinyrenderer` repository by Dmitry V. Sokolov. This repository is a simple 3D renderer that does not rely on any third-party graphics libraries. It is a small codebase that will act as a playground for experimenting with different optimization techniques.

First, we will use a tool like `gperf` to determine the bottlenecks. We will take these results and apply principles from the last lab to optimize. This involves designing an AST to represent the code to optimize (likely loops), implementing generalized schedules (AST to AST) to transform and optimize. This may involve tasks like loop transformations and SIMD vectorization. These optimizations will be performed with our specific systems in mind. Finally we will directly compare the performance of the original code with our transformed code.

The next items on the agenda are to take PC ports of Quake and Super Mario 64 and apply almost an identical process from `tinyrenderer` to each of these games. We will likely want to record a standard set of inputs for each game for when we are comparing performance of the original code versus our optimizations.

For everyone to have a strong foundation it might be a good idea for everyone to do something for `tinyrenderer`. Then we could branch off to working on Quake and Super Mario 64.
