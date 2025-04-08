"""
LEPOR Text Evaluator
This module implements a LEPOR-based text evaluation system for comparing generated text
against a given style input. LEPOR (Language-independent Evaluation Metric for Machine Translation
with Reinforced Factors) is adapted here for style comparison.
"""

import re
import math
import logging
from typing import Dict, List, Tuple, Optional, Union
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class LEPOREvaluator:
    """
    LEPOR-based text evaluator for comparing generated text against a given style input.
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the LEPOR evaluator.
        
        Args:
            language: The language for text evaluation (default: 'english')
        """
        self.language = language
        self.stopwords = set(stopwords.words(language))
        
        # LEPOR parameters (can be tuned)
        self.alpha = 0.25  # Length penalty weight
        self.beta = 0.25   # Position penalty weight
        self.gamma = 0.25  # N-gram precision weight
        self.delta = 0.25  # Word order weight
        
        logger.info(f"LEPOR evaluator initialized for {language}")
    
    def evaluate(self, generated_text: str, style_reference: str) -> Dict[str, float]:
        """
        Evaluate generated text against a style reference using LEPOR metrics.
        
        Args:
            generated_text: The text to evaluate
            style_reference: The reference style text
            
        Returns:
            Dictionary containing LEPOR scores and component scores
        """
        # Tokenize texts
        gen_tokens = word_tokenize(generated_text.lower())
        ref_tokens = word_tokenize(style_reference.lower())
        
        # Calculate component scores
        length_score = self._length_penalty(gen_tokens, ref_tokens)
        position_score = self._position_penalty(gen_tokens, ref_tokens)
        ngram_score = self._n_gram_precision(gen_tokens, ref_tokens)
        word_order_score = self._word_order_similarity(gen_tokens, ref_tokens)
        
        # Calculate final LEPOR score
        lepor_score = (
            self.alpha * length_score +
            self.beta * position_score +
            self.gamma * ngram_score +
            self.delta * word_order_score
        )
        
        # Return all scores
        return {
            "lepor_score": lepor_score,
            "length_penalty": length_score,
            "position_penalty": position_score,
            "n_gram_precision": ngram_score,
            "word_order_similarity": word_order_score
        }
    
    def _length_penalty(self, gen_tokens: List[str], ref_tokens: List[str]) -> float:
        """
        Calculate length penalty score.
        
        Args:
            gen_tokens: Tokens from generated text
            ref_tokens: Tokens from reference text
            
        Returns:
            Length penalty score between 0 and 1
        """
        gen_len = len(gen_tokens)
        ref_len = len(ref_tokens)
        
        if ref_len == 0:
            return 0.0
        
        # Calculate brevity penalty
        if gen_len < ref_len:
            return math.exp(1 - ref_len / gen_len) if gen_len > 0 else 0.0
        else:
            return math.exp(1 - gen_len / ref_len)
    
    def _position_penalty(self, gen_tokens: List[str], ref_tokens: List[str]) -> float:
        """
        Calculate position penalty score.
        
        Args:
            gen_tokens: Tokens from generated text
            ref_tokens: Tokens from reference text
            
        Returns:
            Position penalty score between 0 and 1
        """
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        # Create word position dictionaries
        gen_positions = {}
        ref_positions = {}
        
        for i, token in enumerate(gen_tokens):
            if token not in gen_positions:
                gen_positions[token] = []
            gen_positions[token].append(i)
        
        for i, token in enumerate(ref_tokens):
            if token not in ref_positions:
                ref_positions[token] = []
            ref_positions[token].append(i)
        
        # Calculate position similarity
        position_scores = []
        for token in set(gen_tokens) & set(ref_tokens):  # Common tokens
            gen_pos = gen_positions[token]
            ref_pos = ref_positions[token]
            
            # Calculate minimum distance between positions
            min_dist = float('inf')
            for g_pos in gen_pos:
                for r_pos in ref_pos:
                    dist = abs(g_pos - r_pos)
                    min_dist = min(min_dist, dist)
            
            # Normalize distance
            max_len = max(len(gen_tokens), len(ref_tokens))
            if max_len > 0:
                position_scores.append(1.0 - (min_dist / max_len))
        
        return sum(position_scores) / len(position_scores) if position_scores else 0.0
    
    def _n_gram_precision(self, gen_tokens: List[str], ref_tokens: List[str], 
                          max_n: int = 4) -> float:
        """
        Calculate n-gram precision score.
        
        Args:
            gen_tokens: Tokens from generated text
            ref_tokens: Tokens from reference text
            max_n: Maximum n-gram size
            
        Returns:
            N-gram precision score between 0 and 1
        """
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        # Calculate n-gram precision for different n values
        ngram_scores = []
        
        for n in range(1, min(max_n + 1, len(gen_tokens) + 1)):
            # Generate n-grams
            gen_ngrams = self._get_ngrams(gen_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            
            # Count matches
            matches = sum(1 for ngram in gen_ngrams if ngram in ref_ngrams)
            
            # Calculate precision
            precision = matches / len(gen_ngrams) if gen_ngrams else 0.0
            ngram_scores.append(precision)
        
        # Return geometric mean of n-gram precisions
        if ngram_scores:
            return math.exp(sum(math.log(score) if score > 0 else float('-inf') 
                               for score in ngram_scores) / len(ngram_scores))
        return 0.0
    
    def _word_order_similarity(self, gen_tokens: List[str], ref_tokens: List[str]) -> float:
        """
        Calculate word order similarity score.
        
        Args:
            gen_tokens: Tokens from generated text
            ref_tokens: Tokens from reference text
            
        Returns:
            Word order similarity score between 0 and 1
        """
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        # Create word position dictionaries
        gen_positions = {}
        ref_positions = {}
        
        for i, token in enumerate(gen_tokens):
            if token not in gen_positions:
                gen_positions[token] = []
            gen_positions[token].append(i)
        
        for i, token in enumerate(ref_tokens):
            if token not in ref_positions:
                ref_positions[token] = []
            ref_positions[token].append(i)
        
        # Calculate Kendall's tau for word order
        common_tokens = list(set(gen_tokens) & set(ref_tokens))
        if len(common_tokens) < 2:
            return 0.0
        
        # Create position lists for common tokens
        gen_order = []
        ref_order = []
        
        for token in common_tokens:
            gen_order.append(gen_positions[token][0])  # Use first occurrence
            ref_order.append(ref_positions[token][0])  # Use first occurrence
        
        # Calculate Kendall's tau
        tau = self._kendall_tau(gen_order, ref_order)
        
        # Normalize to [0, 1] range
        return (tau + 1) / 2
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """
        Generate n-grams from a list of tokens.
        
        Args:
            tokens: List of tokens
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _kendall_tau(self, list1: List[int], list2: List[int]) -> float:
        """
        Calculate Kendall's tau rank correlation coefficient.
        
        Args:
            list1: First list of ranks
            list2: Second list of ranks
            
        Returns:
            Kendall's tau value between -1 and 1
        """
        if len(list1) != len(list2):
            return 0.0
        
        # Create rank dictionaries
        rank1 = {val: i for i, val in enumerate(list1)}
        rank2 = {val: i for i, val in enumerate(list2)}
        
        # Count concordant and discordant pairs
        concordant = 0
        discordant = 0
        
        for i in range(len(list1)):
            for j in range(i + 1, len(list1)):
                # Check if pair is concordant or discordant
                if (rank1[list1[i]] < rank1[list1[j]] and 
                    rank2[list1[i]] < rank2[list1[j]]):
                    concordant += 1
                elif (rank1[list1[i]] > rank1[list1[j]] and 
                      rank2[list1[i]] > rank2[list1[j]]):
                    concordant += 1
                else:
                    discordant += 1
        
        # Calculate tau
        total = concordant + discordant
        if total == 0:
            return 0.0
        
        return (concordant - discordant) / total
    
    def analyze_style_features(self, text: str) -> Dict[str, Any]:
        """
        Analyze style features of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing style features
        """
        # Tokenize text
        sentences = sent_tokenize(text)
        tokens = word_tokenize(text.lower())
        
        # Calculate basic statistics
        avg_sentence_length = len(tokens) / len(sentences) if sentences else 0
        avg_word_length = sum(len(token) for token in tokens) / len(tokens) if tokens else 0
        
        # Calculate vocabulary richness
        unique_tokens = set(tokens)
        vocabulary_richness = len(unique_tokens) / len(tokens) if tokens else 0
        
        # Calculate stopword ratio
        stopword_count = sum(1 for token in tokens if token in self.stopwords)
        stopword_ratio = stopword_count / len(tokens) if tokens else 0
        
        # Identify sentence types
        declarative = sum(1 for sent in sentences if sent.strip().endswith('.'))
        interrogative = sum(1 for sent in sentences if sent.strip().endswith('?'))
        exclamatory = sum(1 for sent in sentences if sent.strip().endswith('!'))
        
        # Calculate punctuation density
        punctuation = sum(1 for char in text if char in '.,;:!?')
        punctuation_density = punctuation / len(text) if text else 0
        
        return {
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "vocabulary_richness": vocabulary_richness,
            "stopword_ratio": stopword_ratio,
            "sentence_types": {
                "declarative": declarative,
                "interrogative": interrogative,
                "exclamatory": exclamatory
            },
            "punctuation_density": punctuation_density
        }
    
    def compare_style_features(self, generated_text: str, style_reference: str) -> Dict[str, Any]:
        """
        Compare style features between generated text and style reference.
        
        Args:
            generated_text: Generated text to evaluate
            style_reference: Reference style text
            
        Returns:
            Dictionary containing style comparison results
        """
        # Analyze style features
        gen_features = self.analyze_style_features(generated_text)
        ref_features = self.analyze_style_features(style_reference)
        
        # Calculate feature differences
        differences = {}
        for feature in gen_features:
            if isinstance(gen_features[feature], dict):
                differences[feature] = {
                    k: abs(gen_features[feature].get(k, 0) - ref_features[feature].get(k, 0))
                    for k in set(gen_features[feature].keys()) | set(ref_features[feature].keys())
                }
            else:
                differences[feature] = abs(gen_features[feature] - ref_features[feature])
        
        # Calculate overall style similarity
        style_similarity = 1.0 - sum(differences.values()) / len(differences)
        
        return {
            "generated_features": gen_features,
            "reference_features": ref_features,
            "feature_differences": differences,
            "style_similarity": style_similarity
        }
    
    def evaluate_with_style(self, generated_text: str, style_reference: str) -> Dict[str, Any]:
        """
        Comprehensive evaluation of generated text against style reference.
        
        Args:
            generated_text: Generated text to evaluate
            style_reference: Reference style text
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        # Get LEPOR scores
        lepor_scores = self.evaluate(generated_text, style_reference)
        
        # Get style comparison
        style_comparison = self.compare_style_features(generated_text, style_reference)
        
        # Combine results
        return {
            "lepor_scores": lepor_scores,
            "style_comparison": style_comparison,
            "overall_score": (lepor_scores["lepor_score"] + style_comparison["style_similarity"]) / 2
        } 