"""
Advanced retrieval techniques for the AIRI chatbot.
Implements MMR (Maximal Marginal Relevance) deduplication and cross-encoder re-ranking.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from langchain.docstore.document import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from ...config.logging import get_logger
from ...config.settings import settings

logger = get_logger(__name__)

class MMRDeduplicator:
    """Implements Maximal Marginal Relevance for reducing redundancy in retrieved documents."""
    
    def __init__(self, lambda_param: float = 0.5, similarity_threshold: float = 0.8):
        """
        Initialize MMR deduplicator.
        
        Args:
            lambda_param: Balance between relevance and diversity (0=max diversity, 1=max relevance)
            similarity_threshold: Threshold for considering documents similar
        """
        self.lambda_param = lambda_param
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def deduplicate_documents(self, 
                            documents: List[Document], 
                            query: str, 
                            max_documents: int = 5) -> List[Document]:
        """
        Apply MMR to select diverse, relevant documents.
        
        Args:
            documents: List of retrieved documents
            query: Original user query
            max_documents: Maximum number of documents to return
            
        Returns:
            List of deduplicated documents
        """
        if not documents:
            return []
        
        if len(documents) <= max_documents:
            return documents
        
        try:
            # Extract text content for vectorization
            doc_texts = [doc.page_content for doc in documents]
            all_texts = [query] + doc_texts
            
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            
            # Calculate relevance scores (similarity to query)
            relevance_scores = cosine_similarity(query_vector, doc_vectors)[0]
            
            # Initialize selected documents and remaining candidates
            selected_docs = []
            remaining_indices = list(range(len(documents)))
            
            # Select first document (highest relevance)
            best_idx = np.argmax(relevance_scores)
            selected_docs.append(documents[best_idx])
            remaining_indices.remove(best_idx)
            
            # Iteratively select documents using MMR
            while len(selected_docs) < max_documents and remaining_indices:
                mmr_scores = []
                
                for idx in remaining_indices:
                    # Relevance component
                    relevance = relevance_scores[idx]
                    
                    # Diversity component (maximum similarity to already selected docs)
                    max_similarity = 0
                    for selected_doc in selected_docs:
                        selected_idx = documents.index(selected_doc)
                        similarity = cosine_similarity(
                            doc_vectors[idx:idx+1], 
                            doc_vectors[selected_idx:selected_idx+1]
                        )[0][0]
                        max_similarity = max(max_similarity, similarity)
                    
                    # MMR score
                    mmr_score = (self.lambda_param * relevance - 
                               (1 - self.lambda_param) * max_similarity)
                    mmr_scores.append((idx, mmr_score))
                
                # Select document with highest MMR score
                best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
                selected_docs.append(documents[best_idx])
                remaining_indices.remove(best_idx)
            
            logger.info(f"MMR deduplication: {len(documents)} → {len(selected_docs)} documents")
            return selected_docs
            
        except Exception as e:
            logger.error(f"Error in MMR deduplication: {str(e)}")
            # Fallback: return first max_documents
            return documents[:max_documents]
    
    def calculate_document_similarity(self, doc1: Document, doc2: Document) -> float:
        """Calculate similarity between two documents."""
        try:
            texts = [doc1.page_content, doc2.page_content]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            logger.error(f"Error calculating document similarity: {str(e)}")
            return 0.0

class CrossEncoderReranker:
    """Cross-encoder re-ranking for improved relevance scoring."""
    
    def __init__(self, use_simple_scoring: bool = True):
        """
        Initialize cross-encoder re-ranker.
        
        Args:
            use_simple_scoring: Whether to use simple heuristic scoring instead of ML models
        """
        self.use_simple_scoring = use_simple_scoring
        
        if not use_simple_scoring:
            # In production, you would load a pre-trained cross-encoder model here
            # For now, we'll use simple heuristic scoring
            logger.info("Cross-encoder model not implemented, using heuristic scoring")
    
    def rerank_documents(self, 
                        documents: List[Document], 
                        query: str,
                        original_scores: Optional[List[float]] = None) -> List[Tuple[Document, float]]:
        """
        Re-rank documents using cross-encoder scoring.
        
        Args:
            documents: List of documents to re-rank
            query: Original user query
            original_scores: Original retrieval scores (if available)
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        try:
            scored_documents = []
            
            for i, doc in enumerate(documents):
                if self.use_simple_scoring:
                    score = self._calculate_heuristic_score(doc, query)
                else:
                    # Here you would use a cross-encoder model
                    score = self._calculate_cross_encoder_score(doc, query)
                
                # Combine with original score if available
                if original_scores and i < len(original_scores):
                    # Weighted combination: 70% cross-encoder, 30% original
                    combined_score = 0.7 * score + 0.3 * original_scores[i]
                else:
                    combined_score = score
                
                scored_documents.append((doc, combined_score))
            
            # Sort by score (highest first)
            scored_documents.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Re-ranked {len(documents)} documents")
            return scored_documents
            
        except Exception as e:
            logger.error(f"Error in cross-encoder re-ranking: {str(e)}")
            # Fallback: return documents with original scores
            if original_scores:
                return list(zip(documents, original_scores))
            else:
                return [(doc, 1.0) for doc in documents]
    
    def _calculate_heuristic_score(self, document: Document, query: str) -> float:
        """Calculate relevance score using heuristics."""
        score = 0.0
        query_lower = query.lower()
        content_lower = document.page_content.lower()
        
        # Keyword matching in content (base score)
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        keyword_overlap = len(query_words.intersection(content_words)) / len(query_words)
        score += keyword_overlap * 0.4
        
        # Exact phrase matching (higher weight)
        if query_lower in content_lower:
            score += 0.3
        
        # Title/metadata matching (high weight)
        title = document.metadata.get('title', '').lower()
        domain = document.metadata.get('domain', '').lower()
        
        if any(word in title for word in query_words):
            score += 0.2
        
        if any(word in domain for word in query_words):
            score += 0.15
        
        # Document type boosting
        file_type = document.metadata.get('file_type', '')
        if file_type == 'ai_risk_entry':
            score += 0.1
        elif file_type == 'ai_risk_domain_summary':
            score += 0.05
        
        # Content length penalty for very short or very long docs
        content_length = len(document.page_content)
        if content_length < 100:
            score *= 0.8
        elif content_length > 2000:
            score *= 0.9
        
        return min(score, 1.0)
    
    def _calculate_cross_encoder_score(self, document: Document, query: str) -> float:
        """Calculate relevance score using cross-encoder model (placeholder)."""
        # This would use a pre-trained cross-encoder model in production
        # For now, fallback to heuristic scoring
        return self._calculate_heuristic_score(document, query)

class AdvancedRetriever:
    """Combines MMR deduplication and cross-encoder re-ranking for optimal retrieval."""
    
    def __init__(self, 
                 mmr_lambda: float = 0.6,
                 mmr_similarity_threshold: float = 0.8,
                 enable_mmr: bool = True,
                 enable_reranking: bool = True):
        """
        Initialize advanced retriever.
        
        Args:
            mmr_lambda: MMR lambda parameter for relevance vs diversity balance
            mmr_similarity_threshold: Similarity threshold for MMR
            enable_mmr: Whether to enable MMR deduplication
            enable_reranking: Whether to enable cross-encoder re-ranking
        """
        self.enable_mmr = enable_mmr
        self.enable_reranking = enable_reranking
        
        if enable_mmr:
            self.mmr_deduplicator = MMRDeduplicator(
                lambda_param=mmr_lambda,
                similarity_threshold=mmr_similarity_threshold
            )
        
        if enable_reranking:
            self.cross_encoder = CrossEncoderReranker(use_simple_scoring=True)
    
    def retrieve_and_rerank(self, 
                           documents: List[Document],
                           query: str,
                           original_scores: Optional[List[float]] = None,
                           max_documents: int = 5) -> List[Document]:
        """
        Apply advanced retrieval techniques to improve document selection.
        
        Args:
            documents: Initial set of retrieved documents
            query: Original user query
            original_scores: Original retrieval scores
            max_documents: Maximum number of documents to return
            
        Returns:
            List of optimally selected and ranked documents
        """
        if not documents:
            return []
        
        processed_docs = documents
        
        try:
            # Step 1: Cross-encoder re-ranking (if enabled)
            if self.enable_reranking:
                logger.info("Applying cross-encoder re-ranking")
                scored_docs = self.cross_encoder.rerank_documents(
                    processed_docs, query, original_scores
                )
                processed_docs = [doc for doc, score in scored_docs]
                # Update scores for next step
                rerank_scores = [score for doc, score in scored_docs]
            else:
                rerank_scores = original_scores
            
            # Step 2: MMR deduplication (if enabled)
            if self.enable_mmr:
                logger.info("Applying MMR deduplication")
                processed_docs = self.mmr_deduplicator.deduplicate_documents(
                    processed_docs, query, max_documents
                )
            else:
                processed_docs = processed_docs[:max_documents]
            
            logger.info(f"Advanced retrieval: {len(documents)} → {len(processed_docs)} documents")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error in advanced retrieval: {str(e)}")
            # Fallback to original documents
            return documents[:max_documents]

# Global advanced retriever instance
advanced_retriever = AdvancedRetriever(
    mmr_lambda=0.6,  # Balanced relevance vs diversity
    mmr_similarity_threshold=0.8,
    enable_mmr=True,
    enable_reranking=True
)