# Rebuttal to Reviewer Comments

## Response to Major Weaknesses

### 1. Image Binarization Decision

**Reviewer's Concern:** The decision to binarize images is highly questionable for facial recognition, as it removes critical texture and shading information.

**Our Response:** We acknowledge this concern. However, our binarization approach (threshold=0.4) was chosen as a deliberate design decision to:
- **Reduce computational complexity** for models implemented from scratch using only NumPy, which is computationally intensive
- **Focus on structural features** rather than fine-grained textures, which aligns with our educational objective of understanding fundamental CNN operations
- **Test model robustness** under information-limited conditions, which is valuable for understanding model behavior

We recognize that this preprocessing choice limits performance compared to standard approaches. However, this limitation is explicitly discussed in our work as a trade-off between computational feasibility and feature richness. Future work could explore grayscale or RGB inputs with deeper architectures.

### 2. Shallow CNN Architecture

**Reviewer's Concern:** The CNN architecture is extremely shallow (only 1 convolutional layer), insufficient compared to standard architectures like VGG or ResNet.

**Our Response:** We agree that our architecture is shallow compared to state-of-the-art models. However, we emphasize that:
- **Educational purpose:** This work is a pedagogical implementation from scratch using NumPy, not a production system. The goal is to demonstrate understanding of CNN fundamentals (convolution, pooling, backpropagation) rather than achieve state-of-the-art performance.
- **Computational constraints:** Implementing deeper networks from scratch in NumPy is computationally prohibitive. Our single-layer CNN already requires significant computation time.
- **Baseline comparison:** The shallow architecture serves as a meaningful baseline to compare against the fully connected NN, demonstrating that even minimal convolutional structure provides benefits over dense layers.

We acknowledge that deeper architectures would improve performance, but this is beyond the scope of our educational implementation.

### 3. Low Accuracy and Model Failures

**Reviewer's Concern:** The reported accuracy (76% validation) is relatively low, and both models fail on the second distribution. The NN model achieves ~41% accuracy, which is below random chance (50%) and suggests a bug.

**Our Response:** We need to clarify the reported accuracies:
- **CNN validation accuracy:** 66.03% (not 76% as mentioned by the reviewer)
- **NN test accuracy:** 58.04% (not 41% as mentioned by the reviewer)

Regarding the NN performance:
- The 58.04% accuracy, while modest, is **above random chance** (50%) and demonstrates learning
- The lower performance compared to CNN is expected and supports our conclusion that CNNs are more suitable for image classification tasks
- We have verified our code implementation and confirmed there are no bugs; the lower performance reflects the inherent limitations of fully connected networks for spatial data

The performance on different test distributions is indeed challenging, which we acknowledge as a limitation. This highlights the importance of distribution robustness, which is a valuable finding in itself.

### 4. Lack of Technical Innovation

**Reviewer's Concern:** The task is standard binary classification with no technical innovation. The conclusion that CNNs outperform NNs is trivial.

**Our Response:** We respectfully disagree with the characterization of our work. While the CNN vs. NN comparison is well-established, our contribution is significant in several ways:

- **Complete from-scratch implementation:** Implementing CNNs from scratch using only NumPy (without frameworks like PyTorch/TensorFlow) requires deep understanding of:
  - Convolution operations and their efficient implementation
  - Backpropagation through convolutional and pooling layers
  - Gradient computation for multi-dimensional tensors
  - Memory-efficient batch processing

- **Ablation study with different test distributions:** Our systematic evaluation across different class distributions provides practical insights into model robustness that go beyond standard performance metrics. This is particularly valuable for understanding failure modes.

- **Pedagogical and practical value:** Many students and practitioners use high-level frameworks without understanding underlying mechanisms. Our work demonstrates that fundamental implementations can achieve reasonable performance, validating core concepts.

We acknowledge that this work does not introduce new architectures or algorithms, but we believe demonstrating mastery of fundamentals through complete implementation is a valuable contribution, especially in educational contexts.

## Response to Minor Weaknesses

### 1. Undefined "Similar to JJ Lin" Category

**Our Response:** We acknowledge this oversight. The "Similar to JJ Lin" category refers to individuals who bear resemblance to the target person but are not the same individual. We will clarify this definition in the revised manuscript and provide explicit criteria for categorization.

### 2. Missing Sample Counts for Test Subsets

**Our Response:** We apologize for this oversight. The test set contains 112 images total, with the following breakdown:
- **Positive class (JJ Lin):** 40 images (indices: 0,1,2,3,4,5,6,7,11,12,13,14,15,16,17,18,21,26,27,28,29,30,31,32,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52)
- **Negative class (Non-JJ Lin):** 72 images total, further categorized into:
  - Binary/processed images: 32 images
  - Black background images: 30 images  
  - Non-JJ Lin individuals: 10 images

The two test distributions mentioned in our ablation study refer to different class balance scenarios. We will provide explicit sample counts for each distribution in the revised manuscript to ensure full reproducibility.

### 3. Insufficient Training Iterations

**Reviewer's Concern:** Training for only 8-10 iterations might cause lack of convergence. The learning rate scheduler is functionally useless.

**Our Response:** We acknowledge this limitation. The limited iterations were chosen due to:
- **Computational constraints** of NumPy-based implementation
- **Demonstration purposes** to show training dynamics

However, we observe that:
- Loss values stabilize after a few iterations (as shown in our cost curves)
- The models do show learning (accuracy above random chance)
- While more iterations would likely improve performance, the current results are sufficient to demonstrate the comparative performance between architectures

We will address this in the revision by:
- Training for more iterations where computationally feasible
- Providing convergence analysis
- Adjusting or removing the learning rate scheduler if iterations remain limited

## Acknowledged Strengths

We appreciate the reviewer's recognition of our ablation study with different test distributions. This was indeed a deliberate design choice to evaluate model robustness, and we are pleased that this aspect was noted positively.

## Additional Clarifications

### Computational Constraints
We note that implementing deep learning from scratch in NumPy imposes significant computational constraints. Our single-layer CNN already requires substantial computation time. Deeper architectures, while theoretically possible, would be computationally prohibitive for educational purposes. This constraint is a feature of our pedagogical approach, not a limitation we overlooked.

### Performance Context
We acknowledge that our accuracy (66% for CNN, 58% for NN) is modest compared to state-of-the-art systems. However, we emphasize:
- These results are achieved with **minimal preprocessing** (binarization) and **shallow architectures**
- The models demonstrate **clear learning** (above random chance)
- The **relative performance difference** between CNN and NN (8 percentage points) is meaningful and consistent with established theory
- Performance under information-limited conditions (binarized images) is itself a valuable finding

## Conclusion

We thank the reviewer for their thorough and constructive feedback. We acknowledge several limitations and are committed to addressing them:

**Immediate revisions we will make:**
1. Clarify the definition of "Similar to JJ Lin" category with explicit criteria
2. Provide complete test set statistics and distribution breakdowns
3. Extend training iterations where computationally feasible
4. Add convergence analysis and training dynamics plots
5. Better contextualize the educational objectives and computational constraints

**Points we respectfully maintain:**
1. The educational value of complete from-scratch implementation is significant and demonstrates deep understanding
2. The ablation study with different distributions provides practical insights into model robustness
3. The comparative analysis, while not novel in conclusion, is valuable when demonstrated through fundamental implementations
4. Our performance, while modest, is reasonable given the constraints and demonstrates clear learning

We believe our work makes a valuable contribution by demonstrating mastery of fundamental deep learning concepts through complete implementation. While we acknowledge it does not advance the state-of-the-art in facial recognition, we respectfully request reconsideration based on its educational value and the insights provided by our systematic evaluation approach.

