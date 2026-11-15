# Theorem 1: The O(1/L) Pooling Dilution Bound

## Formal Statement
Let document $D$ consist of $L$ token representations $h_1, \dots, h_L \in \mathbb{R}^d$. Consider an authoritative fact revision at token position $k$, where token representation $h_k$ changes to $h_k^* = h_k + \delta$.
Under standard uniform mean pooling:
$$ar{h} = rac{1}{L} \sum_{i=1}^L h_i \quad \implies \quad ar{h}^* = ar{h} + rac{1}{L} \delta$$

The normalized global cosine shift is bounded by:
$$\Delta_{\cos} = 1 - rac{\langle ar{h}, ar{h}^* angle}{\|ar{h}\| \|ar{h}^*\|} \le rac{\|\delta\|^2}{2 L^2 \|ar{h}\|^2} = \mathcal{O}\left(rac{1}{L^2}ight)$$

## Intuition
As paragraph length $L$ grows (e.g. $L \ge 200$), a critical factual edit (such as changing *'14 calendar days'* to *'30 calendar days'*) causes a negligible perturbation to the 1024-dimensional dense centroid. Consequently, dense bi-encoders cannot reliably rank fresh versions above older versions.
