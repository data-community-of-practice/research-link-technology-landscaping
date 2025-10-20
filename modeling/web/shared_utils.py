"""
Shared utilities for the Research Landscape Analysis multipage app.
Contains common functions, data loading, and configurations used across multiple pages.
"""

import streamlit as st
import sys
import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pathlib

# Add the parent directory to the Python path to import from the modeling package
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Change working directory to parent to ensure relative paths work
os.chdir(parent_dir)

import config

# Utility functions for simplified field filtering
def format_research_field(field: Optional[str]) -> str:
    """Format a research field value for display."""
    if field is None or pd.isna(field) or str(field).strip() == "":
        return "Unspecified"
    return str(field)

def render_page_links():
    st.logo("web/static/media/logo.png", size="large", icon_image="web/static/media/favicon.png")

    """Render navigation page links"""
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7, width="stretch")
    
    col1.page_link(page="pages/categories.py", label="Categories", icon=":material/category:", width="stretch")
    col2.page_link(page="pages/grants.py", label="Grants", icon=":material/library_books:", width="stretch")
    col3.page_link(page="pages/keywords.py", label="Keywords", icon=":material/tag:", width="stretch")
    col4.page_link(page="pages/research_landscape.py", label="Research Landscapes", icon=":material/document_search:", width="stretch")
    col5.page_link(page="pages/organisation.py", label="Organisation", icon=":material/business:", width="stretch")
    col6.page_link(page="pages/comparing_organisations.py", label="Compare Orgs", icon=":material/compare_arrows:", width="stretch")
    col7.page_link(page="pages/raw_data.py", label="Raw Data", icon=":material/table:", width="stretch")

def hash_df(df: pd.DataFrame) -> int:
    """Generate a hash for a DataFrame based on its content."""
    if "id" in df.columns:
        return pd.util.hash_pandas_object(df.id).sum()
    elif "name" in df.columns:
        return pd.util.hash_pandas_object(df.name).sum()

def hash_df_content(df: pd.DataFrame) -> int:
    """Generate a comprehensive hash for a DataFrame including all content.
    
    This hashes the entire DataFrame content including:
    - All rows and columns
    - Index values
    - Shape (to distinguish filtered vs unfiltered data)
    
    This is more expensive than hash_df but ensures filtered DataFrames
    get different cache keys than their unfiltered versions.
    """
    if df is None or df.empty:
        return hash(None)
    
    # Combine multiple aspects for robust hashing
    shape_hash = hash(df.shape)
    
    # Hash index
    index_hash = pd.util.hash_pandas_object(df.index).sum()
    
    # Hash all columns - handle different types
    column_hashes = []
    for col in df.columns:
        try:
            col_hash = pd.util.hash_pandas_object(df[col], index=False).sum()
            column_hashes.append(col_hash)
        except TypeError:
            # Handle unhashable types (like lists) by converting to string
            col_hash = pd.util.hash_pandas_object(df[col].astype(str), index=False).sum()
            column_hashes.append(col_hash)
    
    content_hash = hash(tuple(column_hashes))
    
    # Combine all hashes
    return hash((shape_hash, index_hash, content_hash))

# Load data with caching
@st.cache_data(hash_funcs={pd.core.frame.DataFrame: hash_df})
def load_data():
    """Load and cache the data"""
    try:
        keywords = config.Keywords.load()
        keywords = keywords[["name", "type", "description", "grants", "organisation_ids"]]
        keywords = keywords.set_index("name")
        grants = config.Grants.load()
        grants = grants[["id", "title", "grant_summary", "funder", "funding_amount", "source", "start_year", "end_year", "organisation_ids"]]
        grants = grants.set_index("id")
        categories = config.Categories.load()
        categories = categories[["name", "description", "field_of_research", "keywords"]]
        categories = categories.set_index("name")
        return keywords, grants, categories
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def load_field_display_names():
    """Load and create FOR codes to name mapping"""
    try:
        with open('data/for_codes_cleaned.json', 'r') as f:
            for_codes_data = json.load(f)
        
        field_display_names = {}
        for code_str, details in for_codes_data.items():
            if isinstance(details, dict) and 'name' in details:
                # Map from FOR code to display name for the dropdown
                field_display_names[code_str] = details['name']
        
        return field_display_names
    except Exception as e:
        st.error(f"Error loading FOR codes data: {e}")
        return {}

@st.cache_data(hash_funcs={pd.core.frame.DataFrame: hash_df})
def get_unique_funders(grants):
    """Get sorted list of unique funders from grants data"""
    return sorted(grants['funder'].dropna().unique())

@st.cache_data(hash_funcs={pd.core.frame.DataFrame: hash_df})
def get_unique_sources(grants):
    """Get sorted list of unique sources from grants data"""
    return sorted(grants['source'].dropna().unique())

@st.cache_data(hash_funcs={pd.core.frame.DataFrame: hash_df})   
def get_unique_keyword_types(keywords):
    """Get sorted list of unique keyword types from keywords data"""
    return sorted(keywords['type'].dropna().unique())

@st.cache_data(hash_funcs={pd.core.frame.DataFrame: hash_df})
def get_unique_research_fields_from_categories(categories):
    """Get sorted list of unique research fields from categories data"""
    if categories is not None and 'field_of_research' in categories.columns:
        return sorted(categories['field_of_research'].dropna().unique())
    return []

@st.cache_data
def get_unique_research_fields():
    """Get unique research subjects from grants data - cached computation"""
    try:
        _, grants, _ = load_data()
        if grants is None or 'primary_subject' not in grants.columns:
            return set()
        subjects = grants['primary_subject'].dropna().unique()
        return set(str(subject) for subject in subjects if str(subject).strip())
    except Exception as e:
        st.error(f"Error getting unique research subjects: {e}")
        return set()


# Vectorized data processing functions
@st.cache_data(hash_funcs={pd.DataFrame: hash_df_content})
def get_category_grant_links(categories_df, keywords_df):
    """Extract category-grant relationships using vectorized pandas operations
    
    Returns DataFrame with columns: category, grant_id
    
    Uses comprehensive content hashing to differentiate filtered DataFrames.
    """
    if categories_df is None or keywords_df is None:
        return pd.DataFrame()
    
    # Reset index to get category names as a column
    cats = categories_df.reset_index()
    if 'name' not in cats.columns:
        cats = cats.rename(columns={cats.columns[0]: 'name'})
    
    # Explode keywords to get one row per category-keyword pair
    cat_kw_pairs = cats[['name', 'keywords']].explode('keywords')
    cat_kw_pairs = cat_kw_pairs.dropna(subset=['keywords'])
    cat_kw_pairs = cat_kw_pairs.rename(columns={'name': 'category', 'keywords': 'keyword'})
    
    if cat_kw_pairs.empty:
        return pd.DataFrame()
    
    # Prepare keywords dataframe
    if 'name' in keywords_df.columns:
        kw_grants = keywords_df[['name', 'grants']].rename(columns={'name': 'keyword'})
    else:
        kw_grants = keywords_df.reset_index()[['name', 'grants']].rename(columns={'name': 'keyword'})
    
    # Merge category-keyword pairs with keyword-grants
    merged = cat_kw_pairs.merge(kw_grants, on='keyword', how='inner')
    
    if merged.empty:
        return pd.DataFrame()
    
    # Explode grants to get one row per category-grant pair
    result = merged[['category', 'grants']].explode('grants')
    result = result.dropna(subset=['grants'])
    result = result.rename(columns={'grants': 'grant_id'})
    
    # CRITICAL: Remove duplicates - a grant can appear in multiple keywords within same category
    result = result.drop_duplicates(subset=['category', 'grant_id'])
    
    return result[['category', 'grant_id']].reset_index(drop=True)


@st.cache_data(hash_funcs={pd.DataFrame: hash_df_content})
def get_keyword_grant_links(keywords_df):
    """Extract keyword-grant relationships using vectorized pandas operations
    
    Returns DataFrame with columns: keyword, grant_id
    
    Uses comprehensive content hashing to differentiate filtered DataFrames.
    """
    if keywords_df is None or keywords_df.empty:
        return pd.DataFrame()
    
    # Prepare keywords with name column
    if 'name' in keywords_df.columns:
        kw_data = keywords_df[['name', 'grants']].rename(columns={'name': 'keyword'})
    else:
        kw_data = keywords_df.reset_index()[['name', 'grants']].rename(columns={'name': 'keyword'})
    
    # Explode grants to get one row per keyword-grant pair
    result = kw_data.explode('grants')
    result = result.dropna(subset=['grants'])
    result = result.rename(columns={'grants': 'grant_id'})
    
    # CRITICAL: Remove duplicates - though rare, a grant might appear multiple times in same keyword
    result = result.drop_duplicates(subset=['keyword', 'grant_id'])
    
    return result[['keyword', 'grant_id']].reset_index(drop=True)


@st.cache_data(hash_funcs={pd.DataFrame: hash_df_content})
def expand_grants_to_years(grants_df, use_active_period=True, include_org_ids=False):
    """Expand grants to one row per year (and optionally per organisation) using vectorized operations
    
    Args:
        grants_df: DataFrame with grant data (must have start_year, end_year columns)
        use_active_period: If True, expand to all years between start and end. If False, only use start year.
        include_org_ids: If True, also expand by organisation_ids to get one row per year per org
        
    Returns:
        DataFrame with columns: grant_id, year, funding_amount, [organisation_id if include_org_ids=True]
    
    Uses comprehensive content hashing to differentiate filtered DataFrames.
    """
    if grants_df.empty:
        return pd.DataFrame()
    
    # Filter grants with valid start_year
    valid_grants = grants_df[grants_df['start_year'].notna()].copy()
    if valid_grants.empty:
        return pd.DataFrame()
    
    # Fill end_year with start_year if missing
    valid_grants['end_year'] = valid_grants['end_year'].fillna(valid_grants['start_year'])
    valid_grants['start_year'] = valid_grants['start_year'].astype(int)
    valid_grants['end_year'] = valid_grants['end_year'].astype(int)
    
    if use_active_period:
        # Create year ranges for each grant
        valid_grants['years'] = valid_grants.apply(
            lambda row: list(range(row['start_year'], row['end_year'] + 1)),
            axis=1
        )
        # Calculate duration and divide funding by duration
        valid_grants['duration'] = valid_grants['end_year'] - valid_grants['start_year'] + 1
        valid_grants['funding_per_year'] = valid_grants['funding_amount'] / valid_grants['duration']
    else:
        # Only use start year
        valid_grants['years'] = valid_grants['start_year'].apply(lambda x: [x])
        valid_grants['funding_per_year'] = valid_grants['funding_amount']
    
    # Explode by years
    expanded = valid_grants[['years', 'funding_per_year']].explode('years')
    expanded = expanded.rename(columns={'years': 'year', 'funding_per_year': 'funding_amount'})
    expanded['grant_id'] = expanded.index
    
    if include_org_ids and 'organisation_ids' in grants_df.columns:
        # Add organisation_ids before exploding
        expanded = expanded.merge(
            grants_df[['organisation_ids']], 
            left_on='grant_id', 
            right_index=True, 
            how='left'
        )
        # Explode by organisation_ids
        expanded = expanded.explode('organisation_ids')
        expanded = expanded.dropna(subset=['organisation_ids'])
        expanded = expanded.rename(columns={'organisation_ids': 'organisation_id'})
        return expanded[['grant_id', 'year', 'organisation_id', 'funding_amount']].reset_index(drop=True)
    
    return expanded[['grant_id', 'year', 'funding_amount']].reset_index(drop=True)


@st.cache_data(hash_funcs={pd.DataFrame: hash_df_content})
def expand_links_to_years(link_df, grants_df, entity_col, use_active_period=True, include_source=False):
    """Expand entity-grant links to one row per entity per year using vectorized operations
    
    Args:
        link_df: DataFrame with entity-grant links (columns: entity_col, grant_id)
        grants_df: DataFrame with grant data
        entity_col: Name of the entity column (e.g., 'category', 'keyword')
        use_active_period: If True, expand to all years between start and end
        include_source: If True, include the 'source' column from grants_df
        
    Returns:
        DataFrame with columns: entity_col, year, grant_id, funding_amount, [source if include_source=True]
    
    Uses comprehensive content hashing to differentiate filtered DataFrames.
    """
    if link_df.empty or grants_df.empty:
        return pd.DataFrame()
    
    # Select columns from grants_df
    grant_cols = ['start_year', 'end_year', 'funding_amount']
    if include_source:
        grant_cols.append('source')
    
    # Merge links with grant data
    merged = link_df.merge(
        grants_df[grant_cols],
        left_on='grant_id',
        right_index=True,
        how='inner'
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    # Filter valid years
    merged = merged[merged['start_year'].notna()].copy()
    if merged.empty:
        return pd.DataFrame()
    
    # Fill end_year with start_year if missing
    merged['end_year'] = merged['end_year'].fillna(merged['start_year'])
    merged['start_year'] = merged['start_year'].astype(int)
    merged['end_year'] = merged['end_year'].astype(int)
    
    if use_active_period:
        # Create year ranges
        merged['years'] = merged.apply(
            lambda row: list(range(row['start_year'], row['end_year'] + 1)),
            axis=1
        )
        # Calculate duration and divide funding by duration
        merged['duration'] = merged['end_year'] - merged['start_year'] + 1
        merged['funding_per_year'] = merged['funding_amount'] / merged['duration']
    else:
        # Only use start year
        merged['years'] = merged['start_year'].apply(lambda x: [x])
        merged['funding_per_year'] = merged['funding_amount']
    
    # Explode by years
    result_cols = [entity_col, 'grant_id', 'years', 'funding_per_year']
    if include_source:
        result_cols.append('source')
    
    expanded = merged[result_cols].explode('years')
    expanded = expanded.rename(columns={'years': 'year', 'funding_per_year': 'funding_amount'})
    
    return expanded.reset_index(drop=True)


# Function to load CSS from the 'static' folder
def load_css():
    css_path = pathlib.Path("web/static/css/styles.css")
    with open(css_path) as f:
        st.html(f"<style>{f.read()}</style>")
