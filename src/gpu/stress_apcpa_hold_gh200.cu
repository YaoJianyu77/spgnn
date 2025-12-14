\documentclass[conference]{IEEEtran}
\usepackage{graphicx}
\usepackage{amsmath, amssymb}

\begin{document}

\title{Efficient Large Sparse Matrix Chain Multiplication on GPUs}
\author{Author Name \\
Affiliation \\
Email}
\maketitle

\begin{abstract}
Efficiently multiplying a chain of sparse matrices is critical in many applications such as network analysis and machine learning. Sparse matrix chain multiplication (SMCM) involves determining the optimal order to multiply matrices to minimize computational cost. Existing approaches often rely on sparsity estimators and dynamic programming to find an efficient multiplication order. In this work, we present a two-stage pipeline for SMCM on GPUs, combining a novel row-wise sparsity estimator and optimized GPU kernels to achieve significant performance improvements.
\end{abstract}

\section{Introduction}
Sparse matrices are commonly used to model interactions in networks, recommendation systems, and knowledge graphs. Multiplying a sequence of sparse matrices efficiently is a fundamental operation in many graph analytics and database queries. One key challenge in sparse matrix chain multiplication (SMCM) is that the performance heavily depends on the order in which matrices are multiplied. The goal is to find a multiplication order that minimizes intermediate sparsity and thus reduces computation time.

Finding the optimal order of multiplication is often approached with dynamic programming, similar to the classic matrix chain multiplication problem. However, for sparse matrices, we also need to accurately estimate the sparsity (number of non-zeros) of intermediate products to guide the ordering.

Traditional sparsity estimators can be inaccurate or slow for very large matrices.

In this work, we propose a novel \emph{row-wise sparsity estimator} (RS-estimator) that leverages the structure of sparse matrices to estimate the number of non-zero entries in intermediate products more accurately. Based on the RS-estimator, we design an efficient algorithm to determine a near-optimal multiplication order for a given matrix chain.

\section{Background and Motivation}
In heterogeneous information networks, meta-paths are used to capture composite relationships between different types of nodes. A meta-path can be formed by the multiplication of adjacency matrices corresponding to individual relations. For example, consider adjacency matrices $A$ and $B$ representing two types of edges; their product $C = A \times B$ represents connectivity following the two-step meta-path. Such meta-path multiplications result in large sparse matrices, demonstrating the need for efficient algorithms to handle such large, sparse computations.

\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{fig_meta_path_gnn.pdf}
\caption{Sparse meta-path matrix multiplication in GNNs: constructing a composite adjacency matrix via the product of two edge-type adjacency matrices.}
\label{fig:meta_path}
\end{figure}

\section{System Overview}
Our approach to efficient SMCM on GPUs consists of two main stages: a planning stage on the CPU and an execution stage on the GPU. In the planning stage, we use the RS-estimator and dynamic programming to determine a near-optimal multiplication order. In the execution stage, we perform the actual matrix multiplications on the GPU according to the planned order, using optimized SpGEMM kernels. Figure~\ref{fig:smcm_pipeline} illustrates the overall pipeline.

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{fig_smcm_pipeline.pdf}
\caption{Overview of the two-stage SMCM pipeline, including the planning stage (CPU) and GPU execution stage.}
\label{fig:smcm_pipeline}
\end{figure}

\section{Target Hardware: NVIDIA GH200}
The NVIDIA GH200 Grace Hopper Superchip is an example of a modern high-performance platform suitable for large-scale graph computations. It features a unified memory architecture, which provides a single shared address space for both CPU and GPU, simplifying data movement. Matrices can be accessed by both processors without explicit data transfers, enabling efficient processing of large graph data as a single unified memory pool.

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{fig_gh200_arch.pdf}
\caption{Diagram of the NVIDIA GH200 Grace Hopper Superchip architecture with unified memory.}
\label{fig:gh200_arch}
\end{figure}

\section{GPU SpGEMM Kernel Optimization}
Efficient sparse matrix multiplication on GPUs requires careful tiling and memory management. Our SpGEMM kernel divides the input matrices into smaller tiles that fit in shared memory. Each thread block is responsible for computing one tile of the output matrix by accumulating products of corresponding tiles from the input matrices.

Figure~\ref{fig:kernel_tile} illustrates how a single SpGEMM kernel processes a tile of the output matrix. 

In addition to tiling, the kernel uses warp-level parallelism and dynamic parallelism to handle irregular sparsity patterns and ensure load balance across threads.

\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{fig_spgemm_tile.pdf}
\caption{Illustration of a GPU SpGEMM kernel processing a tile of the output matrix.}
\label{fig:kernel_tile}
\end{figure}

\section{Optimal Ordering via Dynamic Programming}
To determine the multiplication order, we use a dynamic programming (DP) algorithm. Each DP state corresponds to a subchain $M_i \times \cdots \times M_j$ and stores the best cost and the best split point. We estimate the cost of a split by combining the sizes of intermediate results, as predicted by the RS-estimator.

Figure~\ref{fig:dp_tree} shows an example DP tree for a chain of five matrices. The leaves of the tree represent the original matrices, and each internal node represents the result of multiplying its two child subchains. By exploring different splits, the DP algorithm finds the order that minimizes the estimated overall computation.

\begin{figure}[t]
\centering
\includegraphics[width=0.85\linewidth]{fig_dp_tree.pdf}
\caption{Example dynamic programming tree structure for finding an optimal multiplication order of a chain of matrices.}
\label{fig:dp_tree}
\end{figure}

\section{Conclusion}
We have presented a comprehensive approach for efficient large sparse matrix chain multiplication on modern GPUs. By splitting the process into a planning stage with a novel RS-estimator and a GPU execution stage with optimized SpGEMM kernels, our method effectively handles the challenges of sparsity and large data. Future work includes extending this framework to distributed multi-GPU systems and exploring adaptive sparsity estimation techniques.

\end{document}
