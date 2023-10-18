Constrative Learning to the task of precedence retrieval

Basic Idea: Given a Judgment pair and its distance in the citation graph, we made the contrastive pairs. Distance 1 pairs are +ve pairs, and anything beyond that is a negative pair. Based on this, we trained a transformer model to predict the similarity between the judgments. As judgments are long documents we used a hierarchical transformer architecture.

