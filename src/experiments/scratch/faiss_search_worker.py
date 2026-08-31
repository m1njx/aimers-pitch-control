"""
faiss_search_worker.py — 별도 프로세스에서 faiss만 사용(torch 없음).
PyTorch가 임포트된 프로세스 안에서 faiss.search()를 호출하면 macOS arm64에서
세그폴트가 나는 문제(OpenMP/BLAS 라이브러리 충돌로 추정, KMP_DUPLICATE_LIB_OK로도
해결 안 됨)를 우회하기 위해 완전히 별도 프로세스로 분리.

사용법: python3 faiss_search_worker.py <candidates.npy> <queries.npy> <k> <output.npz>
- candidates.npy: (N_cand, D) float32
- queries.npy: (N_query, D) float32
- output.npz: D(distances, N_query x k), I(indices, N_query x k)
"""
import sys
import numpy as np
import faiss

candidates_path, queries_path, k, output_path = sys.argv[1:5]
k = int(k)

xb = np.ascontiguousarray(np.load(candidates_path).astype(np.float32))
xq = np.ascontiguousarray(np.load(queries_path).astype(np.float32))

index = faiss.IndexFlatL2(xb.shape[1])
index.add(xb)
D, I = index.search(xq, k)

np.savez(output_path, D=D, I=I)
print(f"faiss_search_worker: {xb.shape[0]} candidates, {xq.shape[0]} queries, k={k} -> saved to {output_path}")
