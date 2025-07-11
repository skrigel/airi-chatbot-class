"""
Document processing utilities for Excel and other file formats.
Enhanced with stable RID (Repository ID) assignment for citation tracking.
"""
import os
import csv
import pandas as pd
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document

from ...config.logging import get_logger
from ...config.settings import settings
from ..taxonomy.scqa_taxonomy import scqa_manager

logger = get_logger(__name__)

class DocumentProcessor:
    """Handles processing of various document formats with stable RID assignment."""
    
    def __init__(self):
        self.structured_data = []
        self.rid_counter = 1  # Start RID counter at 1
        self.rid_mapping = {}  # Maps content hash to RID for stability
        self.rid_registry_path = settings.DATA_DIR / "rid_registry.json"
        self._load_rid_registry()
    
    def _load_rid_registry(self):
        """Load existing RID registry for stable citations."""
        try:
            if self.rid_registry_path.exists():
                with open(self.rid_registry_path, 'r') as f:
                    registry = json.load(f)
                    self.rid_mapping = registry.get('mappings', {})
                    self.rid_counter = registry.get('counter', 1)
                logger.info(f"Loaded RID registry with {len(self.rid_mapping)} mappings")
            else:
                logger.info("No existing RID registry found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading RID registry: {str(e)}")
            self.rid_mapping = {}
            self.rid_counter = 1
    
    def _save_rid_registry(self):
        """Save RID registry for persistence."""
        try:
            registry = {
                'mappings': self.rid_mapping,
                'counter': self.rid_counter
            }
            with open(self.rid_registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            logger.info(f"Saved RID registry with {len(self.rid_mapping)} mappings")
        except Exception as e:
            logger.error(f"Error saving RID registry: {str(e)}")
    
    def _assign_rid(self, document: Document) -> str:
        """Assign a stable RID to a document based on its content."""
        # Create a stable hash from key document properties
        content_key = f"{document.page_content[:200]}_{document.metadata.get('source', '')}_{document.metadata.get('row', '')}"
        content_hash = hashlib.md5(content_key.encode()).hexdigest()
        
        # Check if we already have a RID for this content
        if content_hash in self.rid_mapping:
            rid = self.rid_mapping[content_hash]
        else:
            # Assign new RID
            rid = f"RID-{self.rid_counter:05d}"
            self.rid_mapping[content_hash] = rid
            self.rid_counter += 1
        
        # Add RID to document metadata
        document.metadata['rid'] = rid
        document.metadata['content_hash'] = content_hash
        
        # Add field-aware search metadata for boosting
        self._add_search_metadata(document)
        
        # Enhance with SCQA taxonomy structure
        enhanced_document = scqa_manager.enhance_document_with_scqa(document)
        
        return rid
    
    def _add_search_metadata(self, document: Document) -> None:
        """Add searchable metadata fields for field-aware boosting."""
        metadata = document.metadata
        
        # Create concatenated search fields for different priorities
        high_priority_fields = []
        medium_priority_fields = []
        low_priority_fields = []
        
        # High priority: titles, domains, categories
        if 'title' in metadata and metadata['title']:
            high_priority_fields.append(str(metadata['title']))
        if 'domain' in metadata and metadata['domain']:
            high_priority_fields.append(str(metadata['domain']))
        if 'risk_category' in metadata and metadata['risk_category']:
            high_priority_fields.append(str(metadata['risk_category']))
        
        # Medium priority: subdomain, specific domain
        if 'subdomain' in metadata and metadata['subdomain']:
            medium_priority_fields.append(str(metadata['subdomain']))
        if 'specific_domain' in metadata and metadata['specific_domain']:
            medium_priority_fields.append(str(metadata['specific_domain']))
        
        # Low priority: other metadata
        if 'sheet' in metadata and metadata['sheet']:
            low_priority_fields.append(str(metadata['sheet']))
        if 'file_type' in metadata and metadata['file_type']:
            low_priority_fields.append(str(metadata['file_type']))
        
        # Store concatenated fields for search boosting
        metadata['search_high_priority'] = ' '.join(high_priority_fields)
        metadata['search_medium_priority'] = ' '.join(medium_priority_fields)
        metadata['search_low_priority'] = ' '.join(low_priority_fields)
        
        # Create a comprehensive search string for hybrid search
        all_searchable = high_priority_fields + medium_priority_fields + low_priority_fields
        metadata['search_all_fields'] = ' '.join(all_searchable)
        
        # Add content excerpt for search preview
        content_preview = document.page_content[:300]
        if len(document.page_content) > 300:
            content_preview += "..."
        metadata['content_preview'] = content_preview
    
    def process_excel_file(self, file_path: Path) -> List[Document]:
        """
        Process Excel files with specialized handling for AI Risk Repository.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            List of Document objects
        """
        documents = []
        try:
            # Load the Excel file
            excel_data = pd.read_excel(file_path, sheet_name=None)
            logger.info(f"Found {len(excel_data)} sheets in Excel file {file_path}")
            
            # Process each sheet with specialized handling
            for sheet_name, df in excel_data.items():
                logger.info(f"Processing sheet: {sheet_name} with {len(df)} rows and {len(df.columns)} columns")
                
                # Skip empty dataframes
                if df.empty:
                    logger.info(f"Skipping empty sheet: {sheet_name}")
                    continue
                
                # Special handling for the main AI Risk Database sheet
                if 'AI Risk Database v3' in sheet_name:
                    documents.extend(self._process_ai_risk_database_sheet(df, sheet_name, file_path))
                elif 'Domain Taxonomy' in sheet_name:
                    documents.extend(self._process_domain_taxonomy_sheet(df, sheet_name, file_path))
                elif 'Causal Taxonomy' in sheet_name:
                    documents.extend(self._process_causal_taxonomy_sheet(df, sheet_name, file_path))
                else:
                    # Generic processing for other sheets
                    documents.extend(self._process_generic_sheet(df, sheet_name, file_path))
        
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {str(e)}")
            # Create a fallback document
            documents.append(self._create_fallback_document(file_path, str(e)))
        
        # Assign stable RIDs to all documents
        for doc in documents:
            self._assign_rid(doc)
        
        # Save the RID registry after processing
        self._save_rid_registry()
        
        logger.info(f"Created {len(documents)} documents with RIDs from Excel file {file_path}")
        return documents
    
    def _process_ai_risk_database_sheet(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[Document]:
        """Process the main AI Risk Database sheet with proper column detection."""
        documents = []
        
        try:
            # Load with proper header (row 2)
            df_proper = pd.read_excel(file_path, sheet_name=sheet_name, header=2)
            logger.info(f"Loaded AI Risk Database with proper headers: {list(df_proper.columns)}")
            
            # Clean up the dataframe
            df_proper = df_proper.dropna(how='all')
            
            # Group documents by domain for better retrieval
            domain_groups = {}
            individual_docs = []
            
            for index, row in df_proper.iterrows():
                # Skip rows with no meaningful content
                if pd.isna(row.get('Title')) and pd.isna(row.get('Description')):
                    continue
                
                # Extract key information
                title = str(row.get('Title', '')).strip() if pd.notna(row.get('Title')) else ''
                domain = str(row.get('Domain', '')).strip() if pd.notna(row.get('Domain')) else 'Unspecified'
                subdomain = str(row.get('Sub-domain', '')).strip() if pd.notna(row.get('Sub-domain')) else ''
                risk_category = str(row.get('Risk category', '')).strip() if pd.notna(row.get('Risk category')) else ''
                description = str(row.get('Description', '')).strip() if pd.notna(row.get('Description')) else ''
                
                # Create comprehensive content for this risk entry
                content_parts = []
                
                if title:
                    content_parts.append(f"Title: {title}")
                
                if domain and domain != 'Unspecified':
                    content_parts.append(f"Domain: {domain}")
                    
                if subdomain:
                    content_parts.append(f"Sub-domain: {subdomain}")
                    
                if risk_category:
                    content_parts.append(f"Risk Category: {risk_category}")
                    
                if row.get('Risk subcategory') and pd.notna(row.get('Risk subcategory')):
                    content_parts.append(f"Risk Subcategory: {str(row['Risk subcategory']).strip()}")
                    
                if description:
                    content_parts.append(f"Description: {description}")
                    
                # Add additional evidence if available
                if row.get('Additional ev.') and pd.notna(row.get('Additional ev.')):
                    additional_ev = str(row['Additional ev.']).strip()
                    if additional_ev:
                        content_parts.append(f"Additional Evidence: {additional_ev}")
                
                # Add other relevant fields
                for field in ['Entity', 'Intent', 'Timing']:
                    if row.get(field) and pd.notna(row.get(field)):
                        value = str(row[field]).strip()
                        if value:
                            content_parts.append(f"{field}: {value}")
                
                if not content_parts:
                    continue  # Skip empty entries
                
                entry_content = "\\n".join(content_parts)
                
                # Determine the most specific domain for categorization
                specific_domain = subdomain if subdomain else domain
                
                # Create individual document
                doc = Document(
                    page_content=entry_content,
                    metadata={
                        "source": str(file_path),
                        "sheet": sheet_name,
                        "row": index,
                        "file_type": "ai_risk_entry",
                        "title": title if title else f"Risk Entry {index}",
                        "domain": domain,
                        "subdomain": subdomain,
                        "risk_category": risk_category,
                        "specific_domain": specific_domain,
                        "entity": str(row.get('Entity', '')).strip() if pd.notna(row.get('Entity')) else '',
                        "intent": str(row.get('Intent', '')).strip() if pd.notna(row.get('Intent')) else '',
                        "timing": str(row.get('Timing', '')).strip() if pd.notna(row.get('Timing')) else ''
                    }
                )
                individual_docs.append(doc)
                
                # Group by specific domain for aggregated documents
                if specific_domain and specific_domain != 'Unspecified':
                    if specific_domain not in domain_groups:
                        domain_groups[specific_domain] = []
                    domain_groups[specific_domain].append({
                        'content': entry_content,
                        'title': title,
                        'index': index
                    })
            
            # Add individual documents
            documents.extend(individual_docs)
            logger.info(f"Created {len(individual_docs)} individual risk entries")
            
            # Create domain-aggregated documents for better retrieval
            for domain_name, entries in domain_groups.items():
                if len(entries) >= 3:  # Only create aggregated docs for domains with multiple entries
                    domain_doc = self._create_domain_summary_document(
                        domain_name, entries, file_path, sheet_name
                    )
                    documents.append(domain_doc)
            
            logger.info(f"Created {len(domain_groups)} domain summary documents")
            
        except Exception as e:
            logger.error(f"Error processing AI Risk Database sheet: {str(e)}")
            # Fallback to generic processing
            documents.extend(self._process_generic_sheet(df, sheet_name, file_path))
        
        return documents
    
    def _create_domain_summary_document(self, domain_name: str, entries: List[Dict], 
                                       file_path: Path, sheet_name: str) -> Document:
        """Create a domain summary document."""
        domain_content = f"AI Risk Domain: {domain_name}\\n\\n"
        domain_content += f"This domain contains {len(entries)} risk entries from the AI Risk Repository:\\n\\n"
        
        # Include first 10 entries in full, then summarize the rest
        for i, entry in enumerate(entries[:10]):
            domain_content += f"Risk Entry {i+1}:\\n{entry['content']}\\n\\n"
        
        if len(entries) > 10:
            domain_content += f"Additional {len(entries) - 10} entries in this domain include:\\n"
            for entry in entries[10:]:
                if entry['title']:
                    domain_content += f"- {entry['title']}\\n"
        
        return Document(
            page_content=domain_content,
            metadata={
                "source": str(file_path),
                "sheet": sheet_name,
                "file_type": "ai_risk_domain_summary",
                "title": f"AI Risk Domain: {domain_name}",
                "domain": domain_name,
                "entry_count": len(entries),
                "specific_domain": domain_name,
                "is_summary": True,
                "summary_type": "domain_aggregation"
            }
        )
    
    def _process_domain_taxonomy_sheet(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[Document]:
        """Process Domain Taxonomy sheets which contain domain definitions."""
        documents = []
        
        try:
            for index, row in df.iterrows():
                row_text = ' '.join([str(val) for val in row.values if pd.notna(val)]).strip()
                
                if not row_text or len(row_text) < 50:
                    continue
                
                # Check if this looks like a domain definition
                if any(keyword in row_text.lower() for keyword in ['socioeconomic', 'employment', 'economic', 'environmental', 'domain', 'subdomain']):
                    doc = Document(
                        page_content=f"Domain Taxonomy Entry:\\n{row_text}",
                        metadata={
                            "source": str(file_path),
                            "sheet": sheet_name,
                            "row": index,
                            "file_type": "domain_taxonomy",
                            "title": f"Domain Definition from {sheet_name}"
                        }
                    )
                    documents.append(doc)
        
        except Exception as e:
            logger.error(f"Error processing domain taxonomy sheet: {str(e)}")
        
        return documents
    
    def _process_causal_taxonomy_sheet(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[Document]:
        """Process Causal Taxonomy sheets."""
        documents = []
        
        try:
            for index, row in df.iterrows():
                row_text = ' '.join([str(val) for val in row.values if pd.notna(val)]).strip()
                
                if not row_text or len(row_text) < 50:
                    continue
                
                doc = Document(
                    page_content=f"Causal Taxonomy Entry:\\n{row_text}",
                    metadata={
                        "source": str(file_path),
                        "sheet": sheet_name,
                        "row": index,
                        "file_type": "causal_taxonomy",
                        "title": f"Causal Factor from {sheet_name}"
                    }
                )
                documents.append(doc)
        
        except Exception as e:
            logger.error(f"Error processing causal taxonomy sheet: {str(e)}")
        
        return documents
    
    def _process_generic_sheet(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[Document]:
        """Generic processing for other sheets."""
        documents = []
        
        try:
            # Try to detect header row
            header_row = self._detect_header_row(df)
            if header_row > 0:
                df.columns = [str(x).strip() if not pd.isna(x) else f"Column_{i}" 
                             for i, x in enumerate(df.iloc[header_row])]
                df = df.iloc[header_row+1:].reset_index(drop=True)
            
            df = df.dropna(how='all')
            
            # Create a single document for the sheet if it has substantial content
            if len(df) > 0:
                try:
                    if len(df) > 50:
                        sheet_content = df.head(50).to_string(index=False, na_rep="")
                        sheet_content += f"\\n\\n[Additional {len(df) - 50} rows not shown]"
                    else:
                        sheet_content = df.to_string(index=False, na_rep="")
                    
                    doc = Document(
                        page_content=f"Content from sheet {sheet_name}:\\n\\n{sheet_content}",
                        metadata={
                            "source": str(file_path),
                            "sheet": sheet_name,
                            "file_type": "excel_sheet",
                            "title": f"Sheet: {sheet_name}"
                        }
                    )
                    documents.append(doc)
                    
                except Exception as format_err:
                    logger.warning(f"Could not format sheet {sheet_name}: {str(format_err)}")
        
        except Exception as e:
            logger.error(f"Error in generic sheet processing: {str(e)}")
        
        return documents
    
    def _detect_header_row(self, df: pd.DataFrame) -> int:
        """Detect the most likely header row in a DataFrame."""
        header_terms = ['risk', 'domain', 'category', 'description', 'source', 
                       'hazard', 'impact', 'type', 'id', 'reference']
        
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            
            # Check if many cells in this row contain header terms
            header_term_count = sum(1 for cell in row if isinstance(cell, str) and 
                                   any(term in cell.lower() for term in header_terms))
            
            if header_term_count >= 2:
                return i
            
            # Check if row has many string values and not many NaNs
            string_ratio = sum(1 for cell in row if isinstance(cell, str)) / len(row)
            nan_ratio = sum(1 for cell in row if pd.isna(cell)) / len(row)
            
            if string_ratio > 0.7 and nan_ratio < 0.3:
                return i
        
        return 0
    
    def _create_fallback_document(self, file_path: Path, error_msg: str) -> Document:
        """Create a fallback document when processing fails."""
        return Document(
            page_content=f"Excel file: {file_path.name}\\n\\nError processing file: {error_msg}",
            metadata={
                "source": str(file_path),
                "file_type": "excel_error",
                "title": f"Error processing {file_path.name}"
            }
        )