
import sys
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoSES:
    """
    Mixture of Stylistic Experts (MoSEs) framework for AI-generated text detection.
    This implementation focuses on the core components: SRR, SAR, and CTE.
    """
    
    def __init__(self, n_prototypes=5, pca_components=32, random_state=42):
        """
        Initialize MoSEs framework.
        
        Args:
            n_prototypes: Number of prototypes for each style category
            pca_components: Number of PCA components for semantic feature compression
            random_state: Random seed for reproducibility
        """
        self.n_prototypes = n_prototypes
        self.pca_components = pca_components
        self.random_state = random_state
        self.srr_data = None
        self.prototypes = None
        self.pca = None
        self.scaler = StandardScaler()
        self.cte_model = None
        logger.info("MoSEs framework initialized")
    
    def extract_linguistic_features(self, texts, proxy_model=None):
        """
        Extract linguistic features from texts.
        Simplified implementation for demonstration purposes.
        
        Args:
            texts: List of text strings
            proxy_model: Placeholder for proxy model (not implemented)
            
        Returns:
            Array of linguistic features
        """
        logger.info("Extracting linguistic features from texts")
        
        features = []
        for text in texts:
            # Simplified feature extraction
            text_length = len(text.split())
            word_count = len(text.split())
            char_count = len(text)
            avg_word_length = char_count / max(word_count, 1)
            
            # Placeholder values for other features
            log_prob_mean = np.random.normal(0, 1)
            log_prob_var = np.random.uniform(0, 1)
            ngram_repetition_2 = np.random.uniform(0, 0.5)
            ngram_repetition_3 = np.random.uniform(0, 0.3)
            type_token_ratio = np.random.uniform(0.2, 0.8)
            
            feature_vector = [
                text_length,
                log_prob_mean,
                log_prob_var,
                ngram_repetition_2,
                ngram_repetition_3,
                type_token_ratio
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_semantic_embeddings(self, texts):
        """
        Extract semantic embeddings from texts.
        Simplified implementation using random embeddings for demonstration.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of semantic embeddings
        """
        logger.info("Extracting semantic embeddings from texts")
        
        # In a real implementation, this would use a pre-trained language model
        # For demonstration, we generate random embeddings
        embedding_dim = 384  # Typical dimension for sentence embeddings
        embeddings = np.random.randn(len(texts), embedding_dim)
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def build_srr(self, texts, labels, styles=None):
        """
        Build Stylistics Reference Repository (SRR).
        
        Args:
            texts: List of text strings
            labels: List of labels (0 for human, 1 for AI-generated)
            styles: List of style categories (optional)
        """
        logger.info("Building Stylistics Reference Repository (SRR)")
        
        if len(texts) != len(labels):
            logger.error("Texts and labels must have the same length")
            sys.exit(1)
        
        # Extract features
        linguistic_features = self.extract_linguistic_features(texts)
        semantic_embeddings = self.extract_semantic_embeddings(texts)
        
        # Apply PCA to semantic embeddings
        self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        semantic_features = self.pca.fit_transform(semantic_embeddings)
        
        # Combine all features
        all_features = np.concatenate([linguistic_features, semantic_features], axis=1)
        
        # Store SRR data
        self.srr_data = {
            'texts': texts,
            'labels': labels,
            'styles': styles,
            'linguistic_features': linguistic_features,
            'semantic_features': semantic_features,
            'all_features': all_features
        }
        
        logger.info(f"SRR built with {len(texts)} samples and {all_features.shape[1]} features")
    
    def create_prototypes(self):
        """
        Create prototypes for Stylistics-Aware Router (SAR).
        Simplified implementation using clustering.
        """
        logger.info("Creating prototypes for Stylistics-Aware Router")
        
        if self.srr_data is None:
            logger.error("SRR must be built before creating prototypes")
            sys.exit(1)
        
        # For demonstration, we use simple k-means style clustering
        # In a real implementation, this would use optimal transport as described in the paper
        features = self.srr_data['all_features']
        
        # Use nearest neighbors to find prototype centers
        n_samples = features.shape[0]
        n_prototypes = min(self.n_prototypes, n_samples)
        
        # Randomly select initial prototypes
        np.random.seed(self.random_state)
        prototype_indices = np.random.choice(n_samples, n_prototypes, replace=False)
        self.prototypes = features[prototype_indices]
        
        logger.info(f"Created {n_prototypes} prototypes")
    
    def sar_router(self, input_text, m=3):
        """
        Stylistics-Aware Router (SAR) - find relevant reference samples.
        
        Args:
            input_text: Input text to route
            m: Number of nearest prototypes to consider
            
        Returns:
            Indices of relevant reference samples
        """
        logger.info(f"Routing input text using SAR with m={m}")
        
        if self.prototypes is None:
            logger.error("Prototypes must be created before routing")
            sys.exit(1)
        
        # Extract features from input text
        linguistic_features = self.extract_linguistic_features([input_text])
        semantic_embeddings = self.extract_semantic_embeddings([input_text])
        semantic_features = self.pca.transform(semantic_embeddings)
        input_features = np.concatenate([linguistic_features, semantic_features], axis=1)
        
        # Find nearest prototypes
        nbrs = NearestNeighbors(n_neighbors=m, algorithm='ball_tree').fit(self.prototypes)
        distances, indices = nbrs.kneighbors(input_features)
        
        # Find samples closest to these prototypes
        relevant_indices = []
        for prototype_idx in indices[0]:
            # Find samples closest to this prototype
            prototype = self.prototypes[prototype_idx:prototype_idx+1]
            nbrs_samples = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(
                self.srr_data['all_features'])
            _, sample_indices = nbrs_samples.kneighbors(prototype)
            relevant_indices.extend(sample_indices[0])
        
        # Remove duplicates
        relevant_indices = list(set(relevant_indices))
        
        logger.info(f"SAR found {len(relevant_indices)} relevant reference samples")
        return relevant_indices
    
    def train_cte(self, discrimination_scores):
        """
        Train Conditional Threshold Estimator (CTE).
        
        Args:
            discrimination_scores: Array of discrimination scores for SRR samples
        """
        logger.info("Training Conditional Threshold Estimator (CTE)")
        
        if self.srr_data is None:
            logger.error("SRR must be built before training CTE")
            sys.exit(1)
        
        # Prepare features and labels
        X = self.srr_data['all_features']
        y = self.srr_data['labels']
        
        # Add discrimination scores as a feature
        X_with_scores = np.column_stack([X, discrimination_scores])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_with_scores)
        
        # Train logistic regression model (CTE)
        self.cte_model = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced',
            max_iter=1000
        )
        
        try:
            self.cte_model.fit(X_scaled, y)
            logger.info("CTE model trained successfully")
        except Exception as e:
            logger.error(f"Failed to train CTE model: {e}")
            sys.exit(1)
    
    def predict(self, input_text, discrimination_score):
        """
        Make prediction for input text.
        
        Args:
            input_text: Input text to classify
            discrimination_score: Discrimination score from base model
            
        Returns:
            Prediction (0 for human, 1 for AI-generated) and confidence
        """
        logger.info(f"Making prediction for input text with score: {discrimination_score}")
        
        if self.cte_model is None:
            logger.error("CTE model must be trained before making predictions")
            sys.exit(1)
        
        # Find relevant reference samples using SAR
        relevant_indices = self.sar_router(input_text)
        
        if not relevant_indices:
            logger.error("No relevant reference samples found")
            sys.exit(1)
        
        # Extract features from input text
        linguistic_features = self.extract_linguistic_features([input_text])
        semantic_embeddings = self.extract_semantic_embeddings([input_text])
        semantic_features = self.pca.transform(semantic_embeddings)
        input_features = np.concatenate([linguistic_features, semantic_features], axis=1)
        
        # Add discrimination score as feature
        input_features_with_score = np.column_stack([input_features, discrimination_score])
        
        # Scale features
        input_scaled = self.scaler.transform(input_features_with_score)
        
        # Make prediction
        try:
            prediction = self.cte_model.predict(input_scaled)[0]
            confidence = self.cte_model.predict_proba(input_scaled)[0].max()
            
            logger.info(f"Prediction: {'AI-generated' if prediction == 1 else 'Human-written'}")
            logger.info(f"Confidence: {confidence:.4f}")
            
            return prediction, confidence
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            sys.exit(1)

def main():
    """
    Main function to demonstrate MoSEs framework.
    """
    logger.info("Starting MoSEs experiment")
    
    # Create sample data for demonstration
    # In a real implementation, this would come from actual datasets
    np.random.seed(42)
    
    # Generate sample texts
    n_samples = 100
    human_texts = [
        f"This is a human-written text about topic {i}. It contains natural language patterns " +
        f"and varied sentence structures that are characteristic of human writing." 
        for i in range(n_samples // 2)
    ]
    
    ai_texts = [
        f"This is an AI-generated text about topic {i}. It demonstrates typical patterns " +
        f"found in machine-generated content with consistent style and structure."
        for i in range(n_samples // 2)
    ]
    
    all_texts = human_texts + ai_texts
    labels = [0] * (n_samples // 2) + [1] * (n_samples // 2)  # 0=human, 1=AI
    
    # Generate sample discrimination scores
    # Human texts tend to have lower scores, AI texts higher scores
    human_scores = np.random.normal(-0.5, 0.7, n_samples // 2)
    ai_scores = np.random.normal(0.5, 0.7, n_samples // 2)
    discrimination_scores = np.concatenate([human_scores, ai_scores])
    
    # Initialize and train MoSEs
    meses = MoSES(n_prototypes=3, pca_components=10)
    
    # Build SRR
    meses.build_srr(all_texts, labels)
    
    # Create prototypes
    meses.create_prototypes()
    
    # Train CTE
    meses.train_cte(discrimination_scores)
    
    # Test with sample texts
    test_texts = [
        "This is a human-written article discussing the implications of artificial intelligence.",
        "The rapid advancement of large language models has intensified public concerns.",
        "As an AI system, I generate text based on patterns learned from training data."
    ]
    
    test_scores = [-0.3, 0.1, 0.8]  # Example discrimination scores
    
    logger.info("\n" + "="*50)
    logger.info("TESTING MoSEs FRAMEWORK")
    logger.info("="*50)
    
    results = []
    for i, (text, score) in enumerate(zip(test_texts, test_scores)):
        logger.info(f"\nTest {i+1}:")
        logger.info(f"Text: {text[:100]}...")
        logger.info(f"Discrimination score: {score:.4f}")
        
        prediction, confidence = meses.predict(text, score)
        results.append((prediction, confidence))
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*50)
    logger.info("MoSEs framework successfully implemented with:")
    logger.info(f"- SRR containing {n_samples} samples")
    logger.info(f"- {meses.n_prototypes} prototypes for SAR")
    logger.info(f"- CTE using logistic regression")
    logger.info("\nTest results:")
    
    for i, (prediction, confidence) in enumerate(results):
        pred_label = "AI-generated" if prediction == 1 else "Human-written"
        logger.info(f"Test {i+1}: {pred_label} (confidence: {confidence:.4f})")
    
    logger.info("\nExperiment completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)
