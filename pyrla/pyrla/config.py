"""
Configuration file for PyRLA

This module contains all configuration constants and magic numbers
used throughout the PyRLA package.
"""


class APIConfig:
    """API-related configuration constants"""
    
    # Request timeouts
    DEFAULT_TIMEOUT = 60  # seconds
    
    # Pagination limits
    DEFAULT_PAGE_SIZE = 10
    MAX_PAGE_SIZE = 250  # API limit
    MAX_PAGE_SIZE_SWAGGER = 50  # API limit for Swagger UI
    
    # Concurrency limits
    DEFAULT_MAX_CONCURRENT = 5

    # Rate-limit retry settings
    RATE_LIMIT_RETRY_ATTEMPTS = 5
    DEFAULT_RETRY_AFTER = 30
    
    # Base URL
    BASE_URL = "https://researchlink.ardc.edu.au"


class CLIConfig:
    """CLI-related configuration constants"""
    
    # Default limits for CLI commands
    DEFAULT_LIMIT = 20
    
    # Progress bar refresh rate
    PROGRESS_REFRESH_RATE = 10  # updates per second


class FilterConfig:
    """Available filter options for different entity types"""
    
    # Researcher filters based on API documentation
    RESEARCHER_FILTERS = {
        'current_organisation': 'currentOrganisationName',
        'country': 'countryCodes', 
        'state': 'stateCodes',
        'for_subject': 'summary.forSubjectCodes',
        'seo_subject': 'summary.seoSubjectCodes',
        'orcid': 'orcid',
        'first_name': 'firstName',
        'last_name': 'lastName'
    }
    
    # Grant filters based on API documentation
    GRANT_FILTERS = {
        'status': 'status',  # Active, Closed, Not yet accepted
        'funder': 'funder',
        'funding_scheme': 'fundingScheme', 
        'funding_amount': 'fundingAmount',
        'country': 'countryCodes',
        'state': 'stateCodes',
        'for_subject': 'summary.forSubjectCodes',
        'seo_subject': 'summary.seoSubjectCodes',
        'topic': 'research-topic'  # Advanced search field
    }
    
    # Organisation filters based on API documentation  
    ORGANISATION_FILTERS = {
        'country': 'countryCodes',
        'state': 'stateCodes', 
        'type': 'type',  # abr or others
        'for_subject': 'summary.forSubjectCodes',
        'seo_subject': 'summary.seoSubjectCodes',
        'topic': 'research-topic'  # Advanced search field
    }


class StatusOptions:
    """Valid status options for grants"""
    GRANT_STATUSES = ['Active', 'Closed', 'Not yet accepted']


class DisplayConfig:
    """Display and formatting configuration"""
    
    # Table truncation limits
    MAX_TITLE_LENGTH = 50
    MAX_DESCRIPTION_LENGTH = 100
    
    # JSON output formatting
    JSON_INDENT = 2
    
    # Console colors (Rich formatting)
    SUCCESS_COLOR = "green"
    ERROR_COLOR = "red" 
    WARNING_COLOR = "yellow"
    INFO_COLOR = "blue"
    DEBUG_COLOR = "dim"
