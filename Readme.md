# CSIRO - Image2Biomass Prediction


#### Dinov3 + Mamba Block Neural Net

```mermaid

graph TD
    %% Input Stage
    Start([Input Full Crop Image]) --> Split[Split Image: Left & Right Halves]
    
    %% Backbone Stage
    subgraph Backbone [Feature Extraction]
    Split --> B1[DINOv3 ViT - Left]
    Split --> B2[DINOv3 ViT - Right]
    B1 --> Cat[Concatenate Tokens: 2N Tokens]
    B2 --> Cat
    end

    %% Fusion Stage
    subgraph Fusion [Sequence Processing]
    Cat --> Mamba[Local Mamba Blocks]
    Mamba --> Pool[Adaptive Avg Pool]
    end

    %% Prediction Heads
    subgraph Heads [Regression & Ratio Heads]
    Pool --> H1[Green Head]
    Pool --> H2[GDM Head]
    Pool --> H3[Total Head]
    Pool --> H4[Dead Head]
    Pool --> H5[Clover Head]
    Pool --> R1[Dead Ratio Head]
    Pool --> R2[Clover Ratio Head]
    end

    %% Logic & Mixing
    subgraph HybridLogic [Physical Constraints]
    H4 & H5 --> Direct[Direct Preds]
    
    %% Derived paths
    H3 & H2 --> D1[Derived Dead: Total - GDM]
    R1 & H3 --> D2[Derived Dead: Ratio * Total]
    H2 & H1 --> D3[Derived Clover: GDM - Green]
    R2 & H2 --> D4[Derived Clover: Ratio * GDM]
    
    D1 & D2 --> DeadAvg[Avg Dead Derived]
    D3 & D4 --> ClovAvg[Avg Clover Derived]
    
    %% Final Mix
    Direct -- Mix Logit --> FinalDead[Final Dead]
    DeadAvg -- Mix Logit --> FinalDead
    
    Direct -- Mix Logit --> FinalClov[Final Clover]
    ClovAvg -- Mix Logit --> FinalClov
    end

    %% Final Output
    H1 --> FinalGreen[Green]
    FinalClov --> FinalGDM[GDM: Green + Clover]
    FinalDead --> FinalTotal[Total: GDM + Dead]

    FinalGreen & FinalDead & FinalClov & FinalGDM & FinalTotal --> Out[/Final Pred5 Tensor/]


```


#### Dinov3 + Classical Models + Mass Balance

```mermaid

graph TD
    %% Input and Embedding
    In([Test Images]) --> Pre[DINOv3 Transform]
    Pre --> Backbone[DINOv3 Backbone<br/>ViT-Huge CLS Token]
    Backbone --> Embed[Embeddings Vector X]

    %% Classical Model Branch
    subgraph ClassicalModels [Parallel Classical Models]
        Embed --> M1[Ridge Pipeline<br/>Scaler + PCA-64]
        Embed --> M2[ElasticNet Pipeline<br/>Scaler + PCA-64]
        Embed --> M3[Linear Reg Pipeline<br/>Scaler + PCA-64]
    end

    %% Ensemble Stage
    M1 --> P1[Preds 1: N,5]
    M2 --> P2[Preds 2: N,5]
    M3 --> P3[Preds 3: N,5]
    
    P1 & P2 & P3 --> Weight[Weighted Ensemble<br/>Weights based on CV Scores]
    
    %% Post-Processing & Physics
    subgraph PostProcessing [Post-Process & Physics Constraint]
        Weight --> Clip1[Clip Negatives: max 0, pred]
        
        Clip1 --> MB[Mass Balance Projection]
        subgraph Math [Projection Math]
            MB -.-> Eq1[Green + Clover = GDM]
            MB -.-> Eq2[GDM + Dead = Total]
        end
        
        MB --> Clip2[Final Non-Negative Clip]
    end

    %% Formatting
    Clip2 --> Pivot[Reformat to Long Format]
    Pivot --> Out[/submission.csv/]

    style MB fill:#f96,stroke:#333,stroke-width:2px
    style Backbone fill:#bbf,stroke:#333



```


#### Final Ensemble

```mermaid

graph TD
    %% Source Streams
    subgraph NN_Branch [Pipeline A: DinoV3 Hybrid NN]
        N1[5-Fold Ensemble] --> N2[Spatial Feature Fusion]
        N2 --> N3[Constraint-Aware Heads]
        N3 --> SUB_A[Submission DF 1]
    end

    subgraph Classical_Branch [Pipeline B: DinoV3 + Classical]
        C1[3-Model Ensemble] --> C2[PCA-64 Embeddings]
        C2 --> C3[Mass Balance Projection]
        C3 --> SUB_B[Submission DF 2]
    end

    %% The Weighted Merge
    SUB_A -- "Weight: 0.65" --> Merge{Weighted Sum}
    SUB_B -- "Weight: 0.35" --> Merge

    %% Processing Logic
    subgraph EnsembleLogic [ensemble_submissions function]
        Merge --> Align[Inner Join on sample_id]
        Align --> Calc[Target = M0*0.65 + M1*0.35]
        Calc --> Float32[Cast to Float32]
    end

    %% Final
    Float32 --> FinalOut[/Final submission.csv/]

    style Merge fill:#ff9,stroke:#333,stroke-width:2px
    style NN_Branch fill:#e1f5fe,stroke:#01579b
    style Classical_Branch fill:#f3e5f5,stroke:#4a148c

```

## ⚖️ Mass Balance Projection Math

We enforce physical consistency by ensuring the predicted biomass components satisfy the law of mass conservation. Given a raw prediction vector $\mathbf{x} \in \mathbb{R}^5$ in the order $[\text{Green}, \text{Clover}, \text{Dead}, \text{GDM}, \text{Total}]$, we define two linear constraints:



1. GDM Balance: $\text{Green} + \text{Clover} - \text{GDM} = 0$

2. Total Balance: $\text{Dead} + \text{GDM} - \text{Total} = 0$


This system is represented as $A\mathbf{x} = 0$, where the constraint matrix $A$ is:

$$A = \begin{bmatrix}
1 & 1 & 0 & -1 & 0 \
0 & 0 & 1 & 1 & -1
\end{bmatrix}$$


To find the corrected vector $\mathbf{x}'$ that is closest to our initial prediction $\mathbf{x}$ (minimizing the $L_2$ distance $\|\mathbf{x}' - \mathbf{x}\|^2$), we use the orthogonal projection onto the null space of $A$:


$$\mathbf{x}' = \mathbf{x} - A^T (AA^T)^{-1} (A\mathbf{x})$$

Where:

- $(A\mathbf{x})$ is the residual vector (the "error" in the physical laws).
- $(AA^T)^{-1}$ is the inverse of the constraint covariance.
- The final result $\mathbf{x}'$ is guaranteed to satisfy $A\mathbf{x}' = 0$, ensuring perfect physical alignment across all five predicted targets.



## Final Submission

- The final solution is a hierarchical ensemble combining a **Deep Learning Hybrid (DINOv3 + Mamba)** and a **Classical Feature-Based Ensemble**, both constrained by biomass physics.

![CSIRO Biomass Inference Pipeline](./assets/csiro-final-submisison-flow.png)

-  🚀 Pipeline Breakdown
    1. **Pipeline A (Neural Network):** Uses a Stereo-split DINOv3 backbone fused with **Local Mamba Blocks** to capture spatial context. It employs a "Mix Logit" head to blend direct regression with derived ratios.
    2. **Pipeline B (Classical):** Extracts fixed DINOv3 CLS embeddings, reduces dimensionality via **PCA-64**, and ensembles Ridge, ElasticNet, and Linear models.
    3. **Mass Balance Projection:** Final predictions are projected onto a subspace that satisfies:
        - $Dry\_Green\_g + Dry\_Clover\_g = GDM\_g$
        - $GDM\_g + Dry\_Dead\_g = Dry\_Total\_g$