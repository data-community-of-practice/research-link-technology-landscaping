"""
Main client for interacting with the Research Link Australia (RLA) API
"""

import os
import httpx
import asyncio
import tenacity
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urljoin
import logging
from .config import APIConfig
from .models import Researcher, Grant, Organisation, Publication, SearchResponse
from .exceptions import (
    RLAError, 
    RLAAuthenticationError, 
    RLANotFoundError, 
    RLARateLimitError,
    RLAValidationError,
    RLAServerError
)


logger = logging.getLogger(__name__)


class RLAClient:
    """
    Main client for interacting with the Research Link Australia API
    
    This client supports both async and synchronous operations:
    - Async methods: Use await with method names (recommended for performance)
    - Sync wrappers: Use method names ending with _sync for backward compatibility
    
    Example async usage:
        from pyrla import RLAClient
        
        async def main():
            async with RLAClient(api_token="your_token_here") as client:
                # Search for researchers asynchronously
                researchers = await client.search_researchers(value="John Smith")
                
                # Get specific researcher by ID
                researcher = await client.get_researcher("researcher_id")
                
                # Search for grants
                grants = await client.search_grants(
                    filter_query="funder:Australian Research Council,status:Active"
                )
        
        import asyncio
        asyncio.run(main())
    
    Example synchronous usage:
        from pyrla import RLAClient
        
        client = RLAClient(api_token="your_token_here")
        
        # Search for researchers synchronously
        researchers = client.search_researchers_sync(value="John Smith")
        
        # Get specific researcher by ID
        researcher = client.get_researcher_sync("researcher_id")
        
        # Alternative: use run_async helper
        researchers = client.run_async(client.search_researchers(value="John Smith"))
    """
    
    def __init__(self, api_token: Optional[str] = None, base_url: str = APIConfig.BASE_URL, debug: bool = False):
        """
        Initialize the RLA client
        
        Args:
            api_token: Bearer token for API authentication. If not provided, 
                      will try to get from RLA_API_TOKEN environment variable
            base_url: Base URL for the API (default: https://researchlink.ardc.edu.au)
            debug: Enable debug output for API requests
        """
        self.base_url = base_url.rstrip('/')
        self.debug = debug or os.getenv('PYRLA_DEBUG', '').lower() in ('1', 'true', 'yes')
        
        # Get token from parameter or environment
        self.api_token = api_token or os.getenv('RLA_API_TOKEN')
        if not self.api_token:
            raise RLAAuthenticationError(
                "API token is required. Provide it via api_token parameter or RLA_API_TOKEN environment variable"
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Default request timeout
        self.timeout = APIConfig.DEFAULT_TIMEOUT
        
        # Async session management
        self._session: Optional[httpx.AsyncClient] = None
    
    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(RLARateLimitError),
        wait=lambda rs: rs.outcome.exception().retry_after * (2 ** (rs.attempt_number - 1)),
        stop=tenacity.stop_after_attempt(APIConfig.RATE_LIMIT_RETRY_ATTEMPTS),
        reraise=True,
    )
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make HTTP request to API endpoint
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response data
            
        Raises:
            RLAAuthenticationError: If authentication fails
            RLANotFoundError: If resource not found
            RLAValidationError: If request validation fails
            RLARateLimitError: If rate limited (HTTP 429)
            RLAServerError: If server error occurs
            RLAError: For other API errors
        """
        url = urljoin(self.base_url, endpoint)
        
        try:
            # Show URL in debug mode only if debug is enabled
            if self.debug:
                if params:
                    # Build full URL with parameters for debug display
                    param_str = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
                    full_url = f"{url}?{param_str}" if param_str else url
                    logger.debug(f"Making request to {full_url}")
                    print(f"[DEBUG] API Request: {full_url}")
                    # Also show curl command with token for easy testing
                    print(f"[DEBUG] curl -H 'Authorization: Bearer {self.api_token}' '{full_url}'")
                else:
                    logger.debug(f"Making request to {url}")
                    print(f"[DEBUG] API Request: {url}")
                    # Also show curl command with token for easy testing
                    print(f"[DEBUG] curl -H 'Authorization: Bearer {self.api_token}' '{url}'")
                
            with httpx.Client(headers=self.headers, timeout=self.timeout) as client:
                response = client.get(url, params=params or {})
            
            # Handle different HTTP status codes
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise RLAAuthenticationError("Authentication failed - check API token")
            elif response.status_code == 404:
                raise RLANotFoundError("Resource not found")
            elif response.status_code == 400:
                raise RLAValidationError(f"Invalid request parameters: {response.text}")
            elif response.status_code == 429:
                body = response.json()
                retry_after = body.get("retry_after", APIConfig.DEFAULT_RETRY_AFTER)
                raise RLARateLimitError(
                    f"Rate limited (HTTP 429): {response.text}",
                    retry_after=retry_after,
                    response_data=body,
                )
            elif 500 <= response.status_code < 600:
                raise RLAServerError(f"Server error ({response.status_code}): {response.text}")
            else:
                raise RLAError(f"API request failed ({response.status_code}): {response.text}")
                
        except httpx.RequestError as e:
            raise RLAError(f"Request failed: {e}")
    
    def test_connection(self) -> bool:
        """
        Test API connection and authentication
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._make_request("/search/grants", {"pageSize": 1})
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def run_async(self, coro):
        """
        Helper method to run async methods synchronously
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Result of the coroutine
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, this won't work
                raise RLAError("Cannot run async method synchronously from within an async context. Use await instead.")
            return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(coro)
    
    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create async session"""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(
                headers=self.headers,
                timeout=self.timeout
            )
        return self._session
    
    async def _close_session(self):
        """Close async session"""
        if self._session and not self._session.is_closed:
            await self._session.aclose()
            self._session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()
    
    def __del__(self):
        """Cleanup session on deletion if still open"""
        if self._session and not self._session.is_closed:
            # Create new event loop if none exists (for cleanup)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule cleanup for later
                    loop.create_task(self._close_session())
                else:
                    # If loop is not running, we can run cleanup directly
                    loop.run_until_complete(self._close_session())
            except RuntimeError:
                # If no event loop exists, create one for cleanup
                asyncio.run(self._close_session())
    
    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(RLARateLimitError),
        wait=lambda rs: rs.outcome.exception().retry_after * (2 ** (rs.attempt_number - 1)),
        stop=tenacity.stop_after_attempt(APIConfig.RATE_LIMIT_RETRY_ATTEMPTS),
        reraise=True,
    )
    async def _make_async_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make async HTTP request to API endpoint
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response data
            
        Raises:
            RLAAuthenticationError: If authentication fails
            RLANotFoundError: If resource not found
            RLARateLimitError: If rate limited (HTTP 429)
            RLAValidationError: If request validation fails
            RLAServerError: If server error occurs
            RLAError: For other API errors
        """
        url = urljoin(self.base_url, endpoint)
        session = await self._get_session()
        
        try:
            # Show URL in debug mode only if debug is enabled
            if self.debug:
                if params:
                    # Build full URL with parameters for debug display
                    param_str = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
                    full_url = f"{url}?{param_str}" if param_str else url
                    logger.debug(f"Making async request to {full_url}")
                    print(f"[DEBUG] Async API Request: {full_url}")
                    # Also show curl command with token for easy testing
                    print(f"[DEBUG] curl -H 'Authorization: Bearer {self.api_token}' '{full_url}'")
                else:
                    logger.debug(f"Making async request to {url}")
                    print(f"[DEBUG] Async API Request: {url}")
                    # Also show curl command with token for easy testing
                    print(f"[DEBUG] curl -H 'Authorization: Bearer {self.api_token}' '{url}'")
                
            response = await session.get(url, params=params or {})
            
            # Handle different HTTP status codes
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise RLAAuthenticationError("Authentication failed - check API token")
            elif response.status_code == 404:
                raise RLANotFoundError("Resource not found")
            elif response.status_code == 400:
                raise RLAValidationError(f"Invalid request parameters: {response.text}")
            elif response.status_code == 429:
                body = response.json()
                retry_after = body.get("retry_after", APIConfig.DEFAULT_RETRY_AFTER)
                raise RLARateLimitError(
                    f"Rate limited (HTTP 429): {response.text}",
                    retry_after=retry_after,
                    response_data=body,
                )
            elif 500 <= response.status_code < 600:
                raise RLAServerError(f"Server error ({response.status_code}): {response.text}")
            else:
                raise RLAError(f"API request failed ({response.status_code}): {response.text}")
                    
        except httpx.RequestError as e:
            raise RLAError(f"Request failed: {e}")
        except Exception as e:
            # Ensure session cleanup on unexpected errors
            if isinstance(e, (RLAError, RLAAuthenticationError, RLANotFoundError, RLARateLimitError, RLAValidationError, RLAServerError)):
                # Re-raise our custom exceptions
                raise
            else:
                # Wrap unexpected exceptions
                raise RLAError(f"Unexpected error during request: {e}")
    
    # Researcher methods (async only)
    
    async def search_researchers(
        self,
        value: str = "",
        filter_query: str = "",
        page_number: int = 1,
        page_size: int = APIConfig.DEFAULT_PAGE_SIZE,
        advanced_search_query: str = ""
    ) -> SearchResponse:
        """
        Search for researchers
        
        Args:
            value: Search value (default: "")
            filter_query: Comma-separated filters like "summary.forSubjectCodes:31,currentOrganisationName:Australian National University"
            page_number: Page number (starts from 1, default: 1)
            page_size: Results per page (max 250, default: 10)
            advanced_search_query: Advanced search options like "research-topic:Biological Sciences,firstName:David"
            
        Returns:
            SearchResponse containing list of Researcher objects
        """
        params = {
            "value": value,
            "filterQuery": filter_query,
            "pageNumber": page_number,
            "pageSize": min(page_size, APIConfig.MAX_PAGE_SIZE),  # API limit
            "advancedSearchQuery": advanced_search_query
        }
        
        # Remove empty parameters
        params = {k: v for k, v in params.items() if v}
        
        data = await self._make_async_request("/search/researchers", params)
        return SearchResponse.from_dict(data, Researcher)
    
    async def get_researcher(self, researcher_id: str) -> Researcher:
        """
        Get a specific researcher by ID
        
        Args:
            researcher_id: The researcher's ID
            
        Returns:
            Researcher object
        """
        data = await self._make_async_request(f"/search/researcher/{researcher_id}")
        return Researcher.from_dict(data)
    
    async def search_all_researchers(
        self,
        value: str = "",
        filter_query: str = "",
        advanced_search_query: str = "",
        max_concurrent: int = APIConfig.DEFAULT_MAX_CONCURRENT
    ) -> SearchResponse:
        """
        Search for all researchers across all pages with concurrent requests
        
        Args:
            value: Search value
            filter_query: Filter query
            advanced_search_query: Advanced search query
            max_concurrent: Maximum concurrent requests (default: 5)
            
        Returns:
            SearchResponse containing all researchers
        """
        max_page_size = APIConfig.MAX_PAGE_SIZE  # API limit
        
        # First request to get total count
        first_response = await self.search_researchers(
            value=value,
            filter_query=filter_query,
            page_number=1,
            page_size=max_page_size,
            advanced_search_query=advanced_search_query
        )
        
        all_researchers = list(first_response.results)
        total_results = first_response.total_results
        total_pages = (total_results + max_page_size - 1) // max_page_size
        
        if total_pages > 1:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def fetch_page(page_num: int) -> List[Researcher]:
                async with semaphore:
                    response = await self.search_researchers(
                        value=value,
                        filter_query=filter_query,
                        page_number=page_num,
                        page_size=max_page_size,
                        advanced_search_query=advanced_search_query
                    )
                    return response.results
            
            # Fetch all remaining pages concurrently
            page_tasks = [fetch_page(page_num) for page_num in range(2, total_pages + 1)]
            page_results = await asyncio.gather(*page_tasks, return_exceptions=True)
            
            # Collect results, skip failed requests
            for result in page_results:
                if not isinstance(result, Exception):
                    all_researchers.extend(result)
                else:
                    logger.warning(f"Failed to fetch page: {result}")
        
        return SearchResponse(
            total_results=len(all_researchers),
            current_page=1,
            from_index=0,
            size=len(all_researchers),
            results=all_researchers
        )
    
    # Grant methods (async only)
    
    async def search_grants(
        self,
        value: str = "",
        filter_query: str = "",
        page_number: int = 1,
        page_size: int = APIConfig.DEFAULT_PAGE_SIZE,
        advanced_search_query: str = ""
    ) -> SearchResponse:
        """
        Search for grants
        
        Args:
            value: Search value (default: "")
            filter_query: Comma-separated filters like "funder:Australian Research Council,status:Active"
            page_number: Page number (starts from 1, default: 1)
            page_size: Results per page (max 250, default: 10)
            advanced_search_query: Advanced search options like "research-topic:Biological Sciences,status:Active"
            
        Returns:
            SearchResponse containing list of Grant objects
        """
        params = {
            "value": value,
            "filterQuery": filter_query,
            "pageNumber": page_number,
            "pageSize": min(page_size, APIConfig.MAX_PAGE_SIZE),  # API limit
            "advancedSearchQuery": advanced_search_query
        }
        
        # Remove empty parameters
        params = {k: v for k, v in params.items() if v}
        
        data = await self._make_async_request("/search/grants", params)
        return SearchResponse.from_dict(data, Grant)
    
    async def get_grant(self, grant_id: str) -> Grant:
        """
        Get a specific grant by ID
        
        Args:
            grant_id: The grant's ID
            
        Returns:
            Grant object
        """
        data = await self._make_async_request(f"/search/grant/{grant_id}")
        return Grant.from_dict(data)
    
    async def search_all_grants(
        self,
        value: str = "",
        filter_query: str = "",
        advanced_search_query: str = "",
        max_concurrent: int = APIConfig.DEFAULT_MAX_CONCURRENT
    ) -> SearchResponse:
        """
        Search for all grants across all pages with concurrent requests
        
        Args:
            value: Search value
            filter_query: Filter query
            advanced_search_query: Advanced search query
            max_concurrent: Maximum concurrent requests (default: 5)
            
        Returns:
            SearchResponse containing all grants
        """
        max_page_size = APIConfig.MAX_PAGE_SIZE  # API limit
        
        # First request to get total count
        first_response = await self.search_grants(
            value=value,
            filter_query=filter_query,
            page_number=1,
            page_size=max_page_size,
            advanced_search_query=advanced_search_query
        )
        
        all_grants = list(first_response.results)
        total_results = first_response.total_results
        total_pages = (total_results + max_page_size - 1) // max_page_size
        
        if total_pages > 1:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def fetch_page(page_num: int) -> List[Grant]:
                async with semaphore:
                    response = await self.search_grants(
                        value=value,
                        filter_query=filter_query,
                        page_number=page_num,
                        page_size=max_page_size,
                        advanced_search_query=advanced_search_query
                    )
                    return response.results
            
            # Fetch all remaining pages concurrently
            page_tasks = [fetch_page(page_num) for page_num in range(2, total_pages + 1)]
            page_results = await asyncio.gather(*page_tasks, return_exceptions=True)
            
            # Collect results, skip failed requests
            for result in page_results:
                if not isinstance(result, Exception):
                    all_grants.extend(result)
                else:
                    logger.warning(f"Failed to fetch page: {result}")
        
        return SearchResponse(
            total_results=len(all_grants),
            current_page=1,
            from_index=0,
            size=len(all_grants),
            results=all_grants
        )
    
    # Organisation methods (async only)
    
    async def search_organisations(
        self,
        value: str = "",
        filter_query: str = "",
        page_number: int = 1,
        page_size: int = APIConfig.DEFAULT_PAGE_SIZE,
        advanced_search_query: str = ""
    ) -> SearchResponse:
        """
        Search for organisations
        
        Args:
            value: Search value (default: "")
            filter_query: Comma-separated filters like "countryCodes:au,stateCodes:act"
            page_number: Page number (starts from 1, default: 1)
            page_size: Results per page (max 250, default: 10)
            advanced_search_query: Advanced search options like "research-topic:Biological Sciences,stateCodes:act"
            
        Returns:
            SearchResponse containing list of Organisation objects
        """
        params = {
            "value": value,
            "filterQuery": filter_query,
            "pageNumber": page_number,
            "pageSize": min(page_size, APIConfig.MAX_PAGE_SIZE),  # API limit
            "advancedSearchQuery": advanced_search_query
        }
        
        # Remove empty parameters
        params = {k: v for k, v in params.items() if v}
        
        data = await self._make_async_request("/search/organisations", params)
        return SearchResponse.from_dict(data, Organisation)
    
    async def get_organisation(self, organisation_id: str) -> Organisation:
        """
        Get a specific organisation by ID
        
        Args:
            organisation_id: The organisation's ID
            
        Returns:
            Organisation object
        """
        data = await self._make_async_request(f"/search/organisation/{organisation_id}")
        return Organisation.from_dict(data)
    
    async def search_all_organisations(
        self,
        value: str = "",
        filter_query: str = "",
        advanced_search_query: str = "",
        max_concurrent: int = APIConfig.DEFAULT_MAX_CONCURRENT
    ) -> SearchResponse:
        """
        Search for all organisations across all pages with concurrent requests
        
        Args:
            value: Search value
            filter_query: Filter query
            advanced_search_query: Advanced search query
            max_concurrent: Maximum concurrent requests (default: 5)
            
        Returns:
            SearchResponse containing all organisations
        """
        max_page_size = APIConfig.MAX_PAGE_SIZE  # API limit
        
        # First request to get total count
        first_response = await self.search_organisations(
            value=value,
            filter_query=filter_query,
            page_number=1,
            page_size=max_page_size,
            advanced_search_query=advanced_search_query
        )
        
        all_orgs = list(first_response.results)
        total_results = first_response.total_results
        total_pages = (total_results + max_page_size - 1) // max_page_size
        
        if total_pages > 1:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def fetch_page(page_num: int) -> List[Organisation]:
                async with semaphore:
                    response = await self.search_organisations(
                        value=value,
                        filter_query=filter_query,
                        page_number=page_num,
                        page_size=max_page_size,
                        advanced_search_query=advanced_search_query
                    )
                    return response.results
            
            # Fetch all remaining pages concurrently
            page_tasks = [fetch_page(page_num) for page_num in range(2, total_pages + 1)]
            page_results = await asyncio.gather(*page_tasks, return_exceptions=True)
            
            # Collect results, skip failed requests
            for result in page_results:
                if not isinstance(result, Exception):
                    all_orgs.extend(result)
                else:
                    logger.warning(f"Failed to fetch page: {result}")
        
        return SearchResponse(
            total_results=len(all_orgs),
            current_page=1,
            from_index=0,
            size=len(all_orgs),
            results=all_orgs
        )
    
    # Publication methods (async only)
    
    async def search_publications(
        self,
        value: str = "",
        filter_query: str = "",
        page_number: int = 1,
        page_size: int = APIConfig.DEFAULT_PAGE_SIZE,
        advanced_search_query: str = "",
        max_concurrent: int = APIConfig.DEFAULT_MAX_CONCURRENT
    ) -> SearchResponse:
        """
        Search for publications by searching researchers and extracting their publications
        
        This method uses async requests to fetch researcher details concurrently, 
        significantly improving performance over the synchronous version.
        
        Args:
            value: Search value (publication title, author keywords, etc.)
            filter_query: Comma-separated filters (not directly applicable to publications)
            page_number: Page number (starts from 1, default: 1)
            page_size: Results per page (max 250, default: 10)
            advanced_search_query: Advanced search options like "publications.title:machine learning"
            max_concurrent: Maximum concurrent researcher fetches (default: 10)
            
        Returns:
            SearchResponse containing Publication objects extracted from researcher records
        """
        try:
            # Strategy selection same as sync version
            if advanced_search_query:
                researcher_params = {
                    "pageNumber": 1,
                    "pageSize": min(20, APIConfig.MAX_PAGE_SIZE),
                    "advancedSearchQuery": advanced_search_query
                }
            elif value:
                researcher_params = {
                    "value": value,
                    "pageNumber": 1,
                    "pageSize": min(20, APIConfig.MAX_PAGE_SIZE),
                }
            else:
                researcher_params = {
                    "pageNumber": 1,
                    "pageSize": min(APIConfig.DEFAULT_PAGE_SIZE, APIConfig.MAX_PAGE_SIZE),
                }
            
            # Remove empty parameters
            researcher_params = {k: v for k, v in researcher_params.items() if v}
            
            # Get researchers (simplified data)
            researcher_data = await self._make_async_request("/search/researchers", researcher_params)
            researcher_response = SearchResponse.from_dict(researcher_data, Researcher)
            
            # Extract publications by fetching full researcher details CONCURRENTLY
            all_publications = []
            max_researchers_to_fetch = min(15, len(researcher_response.results))
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def fetch_researcher_publications(researcher) -> List[Publication]:
                async with semaphore:
                    try:
                        # Fetch full researcher details to get publications
                        full_researcher = await self.get_researcher(researcher.id)
                        
                        publications = []
                        if hasattr(full_researcher, 'publications') and full_researcher.publications:
                            # Filter publications based on search criteria
                            for pub in full_researcher.publications:
                                if self._publication_matches_criteria(pub, value, filter_query):
                                    publications.append(pub)
                        
                        return publications
                    except Exception as e:
                        logger.warning(f"Failed to fetch researcher {researcher.id}: {e}")
                        return []
            
            # Fetch all researcher publications concurrently
            researcher_tasks = [
                fetch_researcher_publications(researcher) 
                for researcher in researcher_response.results[:max_researchers_to_fetch]
            ]
            
            # Wait for all tasks to complete
            publication_results = await asyncio.gather(*researcher_tasks, return_exceptions=True)
            
            # Collect all publications from successful fetches
            for result in publication_results:
                if not isinstance(result, Exception):
                    all_publications.extend(result)
                else:
                    logger.warning(f"Failed to fetch researcher publications: {result}")
            
            # Sort publications by relevance (same as sync version)
            if value:
                def sort_key(pub):
                    title_match = value.lower() in (pub.title or "").lower()
                    year = pub.publication_year or 0
                    return (not title_match, -year)
                all_publications.sort(key=sort_key)
            else:
                all_publications.sort(key=lambda p: -(p.publication_year or 0))
            
            # Implement pagination
            start_idx = (page_number - 1) * page_size
            end_idx = start_idx + page_size
            limited_publications = all_publications[start_idx:end_idx]
            
            # Create a SearchResponse-like structure for publications
            search_response = SearchResponse(
                total_results=len(all_publications),
                current_page=page_number,
                from_index=start_idx,
                size=len(limited_publications),
                results=limited_publications
            )
            
            return search_response
            
        except Exception as e:
            # If search fails, return empty results instead of crashing
            logger.error(f"Async publication search failed: {e}")
            return SearchResponse(
                total_results=0,
                current_page=page_number,
                from_index=0,
                size=0,
                results=[]
            )
    
    async def get_publication(self, publication_id: str) -> Publication:
        """
        Get a specific publication by ID
        
        Note: The RLA API doesn't have a direct publication endpoint.
        This method searches through researchers to find a publication with the given ID.
        
        Args:
            publication_id: The publication's ID
            
        Returns:
            Publication object
            
        Raises:
            RLANotFoundError: If publication not found
        """
        try:
            researcher_data = await self._make_async_request("/search/researchers", {
                "advancedSearchQuery": f"publications.id:{publication_id}",
                "pageSize": APIConfig.MAX_PAGE_SIZE
            })
            
            researcher_response = SearchResponse.from_dict(researcher_data, Researcher)
            
            # Look for the publication in the researcher results
            for researcher in researcher_response.results:
                if hasattr(researcher, 'publications') and researcher.publications:
                    for pub in researcher.publications:
                        if hasattr(pub, 'id') and pub.id == publication_id:
                            return pub
            
            raise RLANotFoundError(f"Publication with ID '{publication_id}' not found")
            
        except RLANotFoundError:
            raise
        except Exception as e:
            raise RLAError(f"Error searching for publication {publication_id}: {e}")
    
    def _publication_matches_criteria(self, publication, value: str, filter_query: str) -> bool:
        """
        Check if a publication matches the search criteria
        
        Args:
            publication: Publication object to check
            value: Search value to match against
            filter_query: Filter criteria (basic implementation)
            
        Returns:
            True if publication matches criteria
        """
        if not value and not filter_query:
            return True
        
        # Convert publication to searchable text
        searchable_fields = []
        if hasattr(publication, 'title') and publication.title:
            searchable_fields.append(publication.title.lower())
        if hasattr(publication, 'abstract') and publication.abstract:
            searchable_fields.append(publication.abstract.lower())
        if hasattr(publication, 'authors_list') and publication.authors_list:
            searchable_fields.append(publication.authors_list.lower())
        if hasattr(publication, 'doi') and publication.doi:
            searchable_fields.append(publication.doi.lower())
        
        search_text = " ".join(searchable_fields)
        
        # Check value match
        if value and value.lower() not in search_text:
            return False
        
        # Basic filter query support (can be expanded)
        if filter_query:
            filters = filter_query.split(",")
            for filter_item in filters:
                if ":" in filter_item:
                    key, filter_value = filter_item.split(":", 1)
                    key = key.strip()
                    filter_value = filter_value.strip().lower()
                    
                    if key == "publicationYear" and hasattr(publication, 'publication_year'):
                        # Check publication year
                        if publication.publication_year and filter_value != str(publication.publication_year):
                            return False
                    elif key == "doi" and hasattr(publication, 'doi'):
                        if not publication.doi or filter_value not in publication.doi.lower():
                            return False
                    # Add more filter criteria as needed
        
        return True
    
    async def search_all_publications(
        self,
        value: str = "",
        filter_query: str = "",
        advanced_search_query: str = "",
        max_concurrent: int = APIConfig.DEFAULT_MAX_CONCURRENT
    ) -> SearchResponse:
        """
        Search for all publications across all researchers with concurrent requests
        
        Args:
            value: Search value (publication title, author keywords, etc.)
            filter_query: Comma-separated filters
            advanced_search_query: Advanced search query
            max_concurrent: Maximum concurrent requests (default: 10)
            
        Returns:
            SearchResponse containing all matching publications
        """
        try:
            # Determine strategy based on search criteria
            if advanced_search_query:
                researcher_params = {
                    "pageNumber": 1,
                    "pageSize": APIConfig.MAX_PAGE_SIZE,
                    "advancedSearchQuery": advanced_search_query
                }
            elif value:
                researcher_params = {
                    "value": value,
                    "pageNumber": 1,
                    "pageSize": APIConfig.MAX_PAGE_SIZE,
                }
            else:
                # General search across researchers
                researcher_params = {
                    "pageNumber": 1,
                    "pageSize": APIConfig.MAX_PAGE_SIZE,
                }
            
            # Remove empty parameters
            researcher_params = {k: v for k, v in researcher_params.items() if v}
            
            # Get all researchers
            all_researchers = []
            page = 1
            
            while True:
                researcher_params["pageNumber"] = page
                researcher_data = await self._make_async_request("/search/researchers", researcher_params)
                researcher_response = SearchResponse.from_dict(researcher_data, Researcher)
                
                if not researcher_response.results:
                    break
                    
                all_researchers.extend(researcher_response.results)
                
                # Check if we've reached the last page
                if page >= (researcher_response.total_results + APIConfig.MAX_PAGE_SIZE - 1) // APIConfig.MAX_PAGE_SIZE:
                    break
                    
                page += 1
            
            # Extract publications by fetching full researcher details CONCURRENTLY
            all_publications = []
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def fetch_researcher_publications(researcher) -> List[Publication]:
                async with semaphore:
                    try:
                        # Fetch full researcher details to get publications
                        full_researcher = await self.get_researcher(researcher.id)
                        
                        publications = []
                        if hasattr(full_researcher, 'publications') and full_researcher.publications:
                            # Filter publications based on search criteria
                            for pub in full_researcher.publications:
                                if self._publication_matches_criteria(pub, value, filter_query):
                                    publications.append(pub)
                        
                        return publications
                    except Exception as e:
                        logger.warning(f"Failed to fetch researcher {researcher.id}: {e}")
                        return []
            
            # Fetch all researcher publications concurrently
            researcher_tasks = [
                fetch_researcher_publications(researcher) 
                for researcher in all_researchers
            ]
            
            # Wait for all tasks to complete
            publication_results = await asyncio.gather(*researcher_tasks, return_exceptions=True)
            
            # Collect all publications from successful fetches
            for result in publication_results:
                if not isinstance(result, Exception):
                    all_publications.extend(result)
                else:
                    logger.warning(f"Failed to fetch researcher publications: {result}")
            
            # Remove duplicates based on publication ID
            seen_ids = set()
            unique_publications = []
            for pub in all_publications:
                if hasattr(pub, 'id') and pub.id:
                    if pub.id not in seen_ids:
                        seen_ids.add(pub.id)
                        unique_publications.append(pub)
                else:
                    # If no ID, add anyway (might be duplicate but we can't tell)
                    unique_publications.append(pub)
            
            # Sort publications by relevance
            if value:
                def sort_key(pub):
                    title_match = value.lower() in (pub.title or "").lower()
                    year = pub.publication_year or 0
                    return (not title_match, -year)
                unique_publications.sort(key=sort_key)
            else:
                unique_publications.sort(key=lambda p: -(p.publication_year or 0))
            
            # Create a SearchResponse for all publications
            search_response = SearchResponse(
                total_results=len(unique_publications),
                current_page=1,
                from_index=0,
                size=len(unique_publications),
                results=unique_publications
            )
            
            return search_response
            
        except Exception as e:
            # If search fails, return empty results instead of crashing
            logger.error(f"Async search all publications failed: {e}")
            return SearchResponse(
                total_results=0,
                current_page=1,
                from_index=0,
                size=0,
                results=[]
            )
    
    # Generic search by ID method
    
    async def get_by_id(self, target_type: str, record_id: str) -> Union[Researcher, Grant, Organisation, Publication]:
        """
        Get a record by ID and target type
        
        Args:
            target_type: Type of record ("researcher", "grant", "organisation", "publication")
            record_id: The record's ID
            
        Returns:
            Researcher, Grant, Organisation, or Publication object based on target_type
        """
        if target_type == "publication":
            # Publications are not directly searchable via the API
            return await self.get_publication(record_id)
        
        data = await self._make_async_request(f"/search/{target_type}/{record_id}")
        
        if target_type == "researcher":
            return Researcher.from_dict(data)
        elif target_type == "grant":
            return Grant.from_dict(data)
        elif target_type == "organisation":
            return Organisation.from_dict(data)
        else:
            raise RLAValidationError(f"Invalid target_type: {target_type}. Must be 'researcher', 'grant', 'organisation', or 'publication'")
    
    # Convenience methods
    
    async def search_researchers_by_name(self, first_name: str = "", last_name: str = "", page_size: int = 10) -> SearchResponse:
        """
        Search researchers by name
        
        Args:
            first_name: First name to search for
            last_name: Last name to search for
            page_size: Number of results per page
            
        Returns:
            SearchResponse containing researchers
        """
        query_parts = []
        if first_name:
            query_parts.append(f"firstName:{first_name}")
        if last_name:
            query_parts.append(f"lastName:{last_name}")
        
        advanced_search_query = ",".join(query_parts)
        
        return await self.search_researchers(
            advanced_search_query=advanced_search_query,
            page_size=page_size
        )
    
    async def search_grants_by_funder(self, funder: str, status: str = "", page_size: int = 10) -> SearchResponse:
        """
        Search grants by funding agency
        
        Args:
            funder: Name of the funding agency
            status: Grant status ('Active', 'Closed', 'Not yet accepted')
            page_size: Number of results per page
            
        Returns:
            SearchResponse containing grants
        """
        filter_parts = [f"funder:{funder}"]
        if status:
            filter_parts.append(f"status:{status}")
        
        filter_query = ",".join(filter_parts)
        
        return await self.search_grants(
            filter_query=filter_query,
            page_size=page_size
        )
    
    async def search_active_grants(self, page_size: int = 10) -> SearchResponse:
        """
        Search for active grants
        
        Args:
            page_size: Number of results per page
            
        Returns:
            SearchResponse containing active grants
        """
        return await self.search_grants(
            filter_query="status:Active",
            page_size=page_size
        )
    
    # Synchronous wrapper methods for backward compatibility
    
    def search_researchers_sync(self, **kwargs) -> SearchResponse:
        """Synchronous wrapper for search_researchers"""
        return self.run_async(self.search_researchers(**kwargs))
    
    def get_researcher_sync(self, researcher_id: str) -> Researcher:
        """Synchronous wrapper for get_researcher"""
        return self.run_async(self.get_researcher(researcher_id))
    
    def search_grants_sync(self, **kwargs) -> SearchResponse:
        """Synchronous wrapper for search_grants"""
        return self.run_async(self.search_grants(**kwargs))
    
    def get_grant_sync(self, grant_id: str) -> Grant:
        """Synchronous wrapper for get_grant"""
        return self.run_async(self.get_grant(grant_id))
    
    def search_organisations_sync(self, **kwargs) -> SearchResponse:
        """Synchronous wrapper for search_organisations"""
        return self.run_async(self.search_organisations(**kwargs))
    
    def get_organisation_sync(self, organisation_id: str) -> Organisation:
        """Synchronous wrapper for get_organisation"""
        return self.run_async(self.get_organisation(organisation_id))
    
    def search_publications_sync(self, **kwargs) -> SearchResponse:
        """Synchronous wrapper for search_publications"""
        return self.run_async(self.search_publications(**kwargs))
    
    def get_publication_sync(self, publication_id: str) -> Publication:
        """Synchronous wrapper for get_publication"""
        return self.run_async(self.get_publication(publication_id))
