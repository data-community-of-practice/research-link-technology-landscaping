"""
Command Line Interface for PyRLA - Python interface for Research Link Australia API

This module provides a Typer-based CLI for interacting with the RLA API.
"""

import traceback
import json
import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List
from dataclasses import asdict
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

from .config import APIConfig, CLIConfig, FilterConfig, StatusOptions, DisplayConfig
from .client import RLAClient
from .models import Researcher, Grant, Organisation, SearchResponse
from .exceptions import RLAError, RLAAuthenticationError
from .utils import get_funding_statistics, filter_grants_by_amount, filter_researchers_by_orcid

logger = logging.getLogger(__name__)

# Main app
app = typer.Typer(
    name="pyrla",
    help="PyRLA - Python interface for Research Link Australia API",
    epilog="Visit https://researchlink.ardc.edu.au/v3/api-docs/public for API documentation."
)

# Create subcommand groups
grants_app = typer.Typer(name="grants", help="Commands for working with grants")
app.add_typer(grants_app, name="grants")

console = Console()

# Global client instance and debug flag
client: Optional[RLAClient] = None
debug_mode: bool = False


def build_filter_query(filter_mapping: dict, **kwargs) -> str:
    """Build a filter query string from individual filter parameters"""
    filters = []
    for param_name, value in kwargs.items():
        if value is not None and value != "":
            if param_name in filter_mapping:
                api_field = filter_mapping[param_name]
                filters.append(f"{api_field}:{value}")
    return ",".join(filters)


def build_advanced_query(**kwargs) -> str:
    """Build an advanced search query string from parameters"""
    queries = []
    for param_name, value in kwargs.items():
        if value is not None and value != "":
            queries.append(f"{param_name}:{value}")
    return ",".join(queries)


def combine_filter_queries(*queries) -> str:
    """Combine multiple filter query strings"""
    non_empty = [q for q in queries if q]
    return ",".join(non_empty)


def save_json_to_file(data: List[dict], filename: str) -> None:
    """Save JSON data to a file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=DisplayConfig.JSON_INDENT, ensure_ascii=False, default=str)
        console.print(f"[{DisplayConfig.SUCCESS_COLOR}]✓ JSON data saved to: {filename}[/{DisplayConfig.SUCCESS_COLOR}]")
    except Exception as e:
        console.print(f"[{DisplayConfig.ERROR_COLOR}]Error saving JSON file: {e}[/{DisplayConfig.ERROR_COLOR}]")
        raise typer.Exit(1)


def save_jsonl_to_file(data: List[dict], filename: str) -> None:
    """Save data as JSON Lines (one JSON object per line)"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        console.print(f"[{DisplayConfig.SUCCESS_COLOR}]✓ JSONL data saved to: {filename}[/{DisplayConfig.SUCCESS_COLOR}]")
    except Exception as e:
        console.print(f"[{DisplayConfig.ERROR_COLOR}]Error saving JSONL file: {e}[/{DisplayConfig.ERROR_COLOR}]")
        raise typer.Exit(1)


def output_jsonl_to_stdout(data: List[dict]) -> None:
    """Output data as JSON Lines to stdout"""
    for record in data:
        typer.echo(json.dumps(record, ensure_ascii=False, default=str))


def format_datetime_value(value: Optional[object], default: str = "N/A") -> str:
    """Format datetime/date objects safely for display"""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value in (None, ""):
        return default
    return str(value)


async def enrich_grants_with_summaries(client: RLAClient, input_records: List[dict]) -> List[dict]:
    """Enrich grant records with grant summaries from the RLA API"""
    enriched_records = []
    semaphore = asyncio.Semaphore(APIConfig.DEFAULT_MAX_CONCURRENT)
    
    async def enrich_single_record(record: dict, index: int):
        async with semaphore:
            try:
                # Try different search strategies to find the grant
                grant = None
                search_attempts = []
                
                # Strategy 1: Try to search by key if it exists
                if 'key' in record and record['key']:
                    key = record['key']
                    search_attempts.append(f"key:{key}")
                    
                    # Try extracting ID from key (e.g., "nhmrc/1047630" -> "1047630")
                    if '/' in key:
                        extracted_id = key.split('/')[-1]
                        search_attempts.append(extracted_id)
                
                # Strategy 2: Try local_id
                if 'local_id' in record and record['local_id']:
                    search_attempts.append(record['local_id'])
                
                # Strategy 3: Try title search
                if 'title' in record and record['title']:
                    search_attempts.append(record['title'][:100])  # Truncate long titles
                
                # Attempt searches
                for search_term in search_attempts:
                    if grant:
                        break
                    
                    try:
                        if search_term.startswith('key:'):
                            # Search by key filter
                            response = await client.search_grants(
                                value="",
                                filter_query=f"key:{search_term[4:]}",
                                page_size=1
                            )
                        elif len(search_term) < 50 and search_term.replace('/', '').replace('-', '').isalnum():
                            # Looks like an ID, try get_grant first
                            try:
                                grant = await client.get_grant(search_term)
                                break
                            except RLAError:
                                # If get_grant fails, try search
                                response = await client.search_grants(
                                    value=search_term,
                                    page_size=5
                                )
                        else:
                            # Search by value
                            response = await client.search_grants(
                                value=search_term,
                                page_size=5
                            )
                        
                        if not grant and response.results:
                            # Find best match based on title similarity or exact match
                            best_match = None
                            if 'title' in record and record['title']:
                                record_title = record['title'].lower().strip()
                                for result in response.results:
                                    result_title = (result.title or result.grant_title or '').lower().strip()
                                    if record_title == result_title:
                                        best_match = result
                                        break
                                    elif record_title in result_title or result_title in record_title:
                                        best_match = result
                            
                            if not best_match:
                                best_match = response.results[0]  # Use first result as fallback
                            
                            grant = best_match
                            
                    except RLAError as e:
                        logger.debug(f"Search attempt '{search_term}' failed: {e}")
                        continue
                
                # Create enriched record
                enriched_record = record.copy()
                if grant and grant.grant_summary:
                    enriched_record['grant_summary'] = grant.grant_summary
                    logger.info(f"Enriched record {index + 1}: Found grant summary")
                else:
                    enriched_record['grant_summary'] = ""
                    logger.warning(f"Record {index + 1}: No grant summary found for '{record.get('title', record.get('key', 'Unknown'))}'")
                
                return enriched_record
                
            except Exception as e:
                logger.error(f"Error processing record {index + 1}: {e}")
                enriched_record = record.copy()
                enriched_record['grant_summary'] = ""
                return enriched_record
    
    # Process all records concurrently
    tasks = [enrich_single_record(record, i) for i, record in enumerate(input_records)]
    enriched_records = await asyncio.gather(*tasks, return_exceptions=False)
    
    return enriched_records


def handle_exception(e: Exception, operation: str = "operation"):
    """Handle exceptions with optional debug information"""
    if debug_mode:
        console.print(f"[red]Error during {operation}:[/red]")
        console.print(f"[red]{type(e).__name__}: {str(e)}[/red]")
        console.print("\n[yellow]Full traceback:[/yellow]")
        console.print(traceback.format_exc())
    else:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Use --debug flag to see full traceback[/dim]")


def get_client() -> RLAClient:
    """Get or initialize the RLA client"""
    global client, debug_mode
    # Reset client if debug mode has changed
    if client is not None and client.debug != debug_mode:
        client = None
    
    if client is None:
        try:
            client = RLAClient(debug=debug_mode)
        except RLAAuthenticationError as e:
            console.print(f"[red]Authentication Error:[/red] {e}")
            console.print("[yellow]Please set your API token using:[/yellow]")
            console.print("export RLA_API_TOKEN='your_token_here'")
            raise typer.Exit(1)
    return client


def run_async(coro):
    """Helper function to run async code in sync context"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an event loop, create a new one
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)


def format_researcher_table(researchers: List[Researcher], title: str = "Researchers") -> Table:
    """Format researchers data as a rich table"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan", min_width=20)
    table.add_column("ORCID", style="green", min_width=15)
    table.add_column("Organisation", style="yellow", min_width=30)
    table.add_column("Publications", justify="right", style="blue")
    table.add_column("Grants", justify="right", style="red")
    
    for researcher in researchers:
        name = researcher.full_name or "N/A"
        orcid = researcher.orcid or "N/A"
        org = ", ".join(researcher.current_organisation_name) if researcher.current_organisation_name else "N/A"
        pubs = str(researcher.summary.publication_count) if researcher.summary and researcher.summary.publication_count is not None else "N/A"
        grants = str(researcher.summary.grant_count) if researcher.summary and researcher.summary.grant_count is not None else "N/A"
        
        table.add_row(name, orcid, org, pubs, grants)
    
    return table


def format_grant_table(grants: List[Grant], title: str = "Grants") -> Table:
    """Format grants data as a rich table"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Title", style="cyan", min_width=40)
    table.add_column("Funder", style="green", min_width=15)
    table.add_column("Status", style="yellow", min_width=10)
    table.add_column("Amount (AUD)", justify="right", style="blue")
    table.add_column("Start Date", style="red")
    
    for grant in grants:
        grant_title = grant.title or grant.grant_title or "N/A"
        title_display = grant_title[:60] + "..." if len(grant_title) > 60 else grant_title
        funder = grant.funder or "N/A"
        status = grant.status or "N/A"
        amount = f"${grant.funding_amount:,.2f}" if grant.funding_amount is not None else "N/A"
        start_date = format_datetime_value(grant.start_date)
        
        table.add_row(title_display, funder, status, amount, start_date)
    
    return table


def format_organisation_table(orgs: List[Organisation], title: str = "Organisations") -> Table:
    """Format organisations data as a rich table"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan", min_width=30)
    table.add_column("ABN", style="green", min_width=15)
    table.add_column("Type", style="yellow", min_width=15)
    table.add_column("Location", style="blue", min_width=20)
    table.add_column("Researchers", justify="right", style="red")
    
    for org in orgs:
        name = org.name or org.organisation_name or "N/A"
        abn = org.abn or "N/A"
        org_type = org.abr_type or "N/A"
        
        # Handle location safely
        location_parts = []
        if org.countries:
            location_parts.append(", ".join(org.countries))
        if org.states:
            location_parts.append(", ".join(org.states))
        location = ", ".join(filter(None, location_parts)) or "N/A"
        
        researcher_count = str(len(org.researcher_ids)) if org.researcher_ids else "0"
        
        table.add_row(name, abn, org_type, location, researcher_count)
    
    return table


@app.command()
def test_connection():
    """Test connection to the RLA API"""
    try:
        client = get_client()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Testing connection...", total=None)
            success = client.test_connection()
            progress.update(task, completed=100)
        
        if success:
            console.print("[green]✓ Successfully connected to RLA API![/green]")
        else:
            console.print("[red]✗ Failed to connect to RLA API[/red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error testing connection: {e}[/red]")
        raise typer.Exit(1)


@app.command("get")
def get_item(
    item_type: str = typer.Argument(..., help="Type of item to get (researcher, grant, organisation)"),
    item_id: str = typer.Argument(..., help="ID of the item to retrieve"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save result to JSON file"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Output JSON Lines to stdout"),
    jsonl_file: Optional[str] = typer.Option(None, "--jsonl-file", help="Save results to JSON Lines file"),
    debug: bool = typer.Option(False, "--debug", help="Show full traceback on errors")
):
    """Get a specific item by type and ID"""
    global debug_mode
    debug_mode = debug
    
    options_provided = sum([json_file is not None, jsonl, jsonl_file is not None])
    if options_provided > 1:
        console.print("[red]Error: --json, --jsonl, and --jsonl-file are mutually exclusive[/red]")
        raise typer.Exit(1)
    
    try:
        client = get_client()
        
        # Normalize item type
        item_type = item_type.lower()
        if item_type.endswith('s'):
            item_type = item_type[:-1]  # Remove plural 's'
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if item_type == "researcher":
                task = progress.add_task("Fetching researcher...", total=None)
                item = client.get_researcher_sync(item_id)
                progress.update(task, completed=100)
                
                data = [asdict(item)]
                if json_file:
                    save_json_to_file(data, json_file)
                elif jsonl:
                    output_jsonl_to_stdout(data)
                elif jsonl_file:
                    save_jsonl_to_file(data, jsonl_file)
                else:
                    table = format_researcher_table([item], f"Researcher Details: {item.full_name}")
                    console.print(table)
                    
            elif item_type == "grant":
                task = progress.add_task("Fetching grant...", total=None)
                item = client.get_grant_sync(item_id)
                progress.update(task, completed=100)
                
                data = [asdict(item)]
                if json_file:
                    save_json_to_file(data, json_file)
                elif jsonl:
                    output_jsonl_to_stdout(data)
                elif jsonl_file:
                    save_jsonl_to_file(data, jsonl_file)
                else:
                    table = format_grant_table([item], f"Grant Details: {item.title}")
                    console.print(table)
                    
            elif item_type == "organisation":
                task = progress.add_task("Fetching organisation...", total=None)
                item = client.get_organisation_sync(item_id)
                progress.update(task, completed=100)
                
                data = [asdict(item)]
                if json_file:
                    save_json_to_file(data, json_file)
                elif jsonl:
                    output_jsonl_to_stdout(data)
                elif jsonl_file:
                    save_jsonl_to_file(data, jsonl_file)
                else:
                    table = format_organisation_table([item], f"Organisation Details: {item.name}")
                    console.print(table)
            else:
                console.print(f"[red]Error: Unknown item type '{item_type}'. Valid types: researcher, grant, organisation[/red]")
                raise typer.Exit(1)
                
    except RLAError as e:
        console.print(f"[red]API Error: {e}[/red]")
        if debug_mode:
            console.print("\n[yellow]Full traceback:[/yellow]")
            console.print(traceback.format_exc())
        raise typer.Exit(1)
    except Exception as e:
        handle_exception(e, "get item")
        raise typer.Exit(1)


@app.command("researchers")
def search_researchers(
    query: str = typer.Argument("", help="Search query for researchers (empty returns all researchers)"),
    limit: int = typer.Option(CLIConfig.DEFAULT_LIMIT, "--limit", "-l", help="Maximum number of results to return"),
    all_results: bool = typer.Option(False, "--all", help="Fetch all results from all pages (ignores --limit)"),
    current_organisation: Optional[str] = typer.Option(None, "--org", help="Filter by current organisation name"),
    country: Optional[str] = typer.Option(None, "--country", help="Filter by country code (e.g., 'au')"),
    state: Optional[str] = typer.Option(None, "--state", help="Filter by state code (e.g., 'nsw')"),
    for_subject: Optional[str] = typer.Option(None, "--for-subject", help="Filter by FOR subject code"),
    seo_subject: Optional[str] = typer.Option(None, "--seo-subject", help="Filter by SEO subject code"),
    first_name: Optional[str] = typer.Option(None, "--first-name", help="Filter by first name"),
    last_name: Optional[str] = typer.Option(None, "--last-name", help="Filter by last name"),
    topic: Optional[str] = typer.Option(None, "--topic", help="Search by research topic"),
    orcid_only: bool = typer.Option(False, "--orcid-only", help="Show only researchers with ORCID"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save results to JSON file"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Output JSON Lines to stdout"),
    jsonl_file: Optional[str] = typer.Option(None, "--jsonl-file", help="Save results to JSON Lines file"),
    debug: bool = typer.Option(False, "--debug", help="Show full traceback on errors")
):
    """Search for researchers in the RLA database"""
    global debug_mode
    debug_mode = debug
    
    options_provided = sum([json_file is not None, jsonl, jsonl_file is not None])
    if options_provided > 1:
        console.print("[red]Error: --json, --jsonl, and --jsonl-file are mutually exclusive[/red]")
        raise typer.Exit(1)
    
    try:
        client = get_client()
        
        # Build filter query from individual parameters
        filter_query = build_filter_query(
            FilterConfig.RESEARCHER_FILTERS,
            current_organisation=current_organisation,
            country=country,
            state=state,
            for_subject=for_subject,
            seo_subject=seo_subject,
            first_name=first_name,
            last_name=last_name
        )
        
        # Build advanced query for topic-based search
        advanced_query = build_advanced_query(topic=topic) if topic else ""
        
        # Handle --all option or limit-based fetching
        if all_results:
            console.print("[yellow]Fetching ALL researchers from all pages...[/yellow]")
            
            async def fetch_all_async():
                async with client:
                    response = await client.search_all_researchers(
                        value=query,
                        filter_query=filter_query,
                        advanced_search_query=advanced_query,
                        max_concurrent=APIConfig.DEFAULT_MAX_CONCURRENT
                    )
                    return response
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching all researchers...", total=None)
                response = run_async(fetch_all_async())
                progress.update(task, completed=100, description=f"Fetched {len(response.results)} researchers")
            
            researchers = response.results
        elif limit > APIConfig.MAX_PAGE_SIZE:
            # For larger limits, use async bulk fetching
            console.print(f"[yellow]Fetching {limit} results using concurrent requests...[/yellow]")
            
            async def fetch_limited_async():
                async with client:
                    # Use search_all but limit results
                    response = await client.search_all_researchers(
                        value=query,
                        filter_query=filter_query,
                        advanced_search_query=advanced_query,
                        max_concurrent=APIConfig.DEFAULT_MAX_CONCURRENT
                    )
                    # Limit results to requested amount
                    response.results = response.results[:limit]
                    return response
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching researchers concurrently...", total=None)
                response = run_async(fetch_limited_async())
                progress.update(task, completed=100, description=f"Fetched {len(response.results)} researchers")
            
            researchers = response.results
        else:
            # Single page fetch
            page_size = min(limit, APIConfig.MAX_PAGE_SIZE)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Searching researchers...", total=None)
                response = client.search_researchers_sync(
                    value=query,
                    filter_query=filter_query,
                    page_number=1,
                    page_size=page_size,
                    advanced_search_query=advanced_query
                )
                progress.update(task, completed=100)
            
            researchers = response.results
        if orcid_only:
            researchers = filter_researchers_by_orcid(researchers)
        
        data = [asdict(r) for r in researchers]
        
        if json_file:
            save_json_to_file(data, json_file)
        elif jsonl:
            output_jsonl_to_stdout(data)
        elif jsonl_file:
            save_jsonl_to_file(data, jsonl_file)
        else:
            # Display table format to stdout
            if all_results:
                info_panel = Panel(
                    f"Found {response.total_results} researchers total\n"
                    f"Showing ALL {len(researchers)} results",
                    title="Search Results",
                    border_style="blue"
                )
            else:
                info_panel = Panel(
                    f"Found {response.total_results} researchers total\n"
                    f"Showing {len(researchers)} results (limit: {limit})",
                    title="Search Results",
                    border_style="blue"
                )
            console.print(info_panel)
            
            if researchers:
                table = format_researcher_table(researchers, f"Researchers: {query}")
                console.print(table)
            else:
                console.print("[yellow]No researchers found.[/yellow]")
                
    except RLAError as e:
        console.print(f"[red]API Error: {e}[/red]")
        if debug_mode:
            console.print("\n[yellow]Full traceback:[/yellow]")
            console.print(traceback.format_exc())
        raise typer.Exit(1)
    except Exception as e:
        handle_exception(e, "researcher search")
        raise typer.Exit(1)



@grants_app.command("search")
def search_grants(
    query: str = typer.Argument("", help="Search query for grants (empty returns all grants)"),
    limit: int = typer.Option(CLIConfig.DEFAULT_LIMIT, "--limit", "-l", help="Maximum number of results to return"),
    all_results: bool = typer.Option(False, "--all", help="Fetch all results from all pages (ignores --limit)"),
    ids: Optional[str] = typer.Option(None, "--ids", help="Get specific grants by comma-separated list of IDs"),
    status: Optional[str] = typer.Option(None, "--status", help=f"Filter by grant status ({'/'.join(StatusOptions.GRANT_STATUSES)})"),
    funder: Optional[str] = typer.Option(None, "--funder", help="Filter by funder name"),
    funding_scheme: Optional[str] = typer.Option(None, "--funding-scheme", help="Filter by funding scheme"),
    funding_amount: Optional[float] = typer.Option(None, "--funding-amount", help="Filter by exact funding amount"),
    country: Optional[str] = typer.Option(None, "--country", help="Filter by country code"),
    state: Optional[str] = typer.Option(None, "--state", help="Filter by state code"),
    for_subject: Optional[str] = typer.Option(None, "--for-subject", help="Filter by FOR subject code"),
    seo_subject: Optional[str] = typer.Option(None, "--seo-subject", help="Filter by SEO subject code"),
    topic: Optional[str] = typer.Option(None, "--topic", help="Search by research topic"),
    min_amount: Optional[float] = typer.Option(None, "--min-amount", help="Minimum funding amount"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save results to JSON file"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Output JSON Lines to stdout"),
    jsonl_file: Optional[str] = typer.Option(None, "--jsonl-file", help="Save results to JSON Lines file"),
    show_stats: bool = typer.Option(False, "--stats", help="Show funding statistics"),
    debug: bool = typer.Option(False, "--debug", help="Show full traceback on errors")
):
    """Search for grants in the RLA database"""
    global debug_mode
    debug_mode = debug
    
    options_provided = sum([json_file is not None, jsonl, jsonl_file is not None])
    if options_provided > 1:
        console.print("[red]Error: --json, --jsonl, and --jsonl-file are mutually exclusive[/red]")
        raise typer.Exit(1)
    
    try:
        client = get_client()
        
        # Handle specific grant IDs
        if ids:
            # Parse comma-separated IDs
            grant_ids = [id.strip() for id in ids.split(',') if id.strip()]
            console.print(f"[yellow]Fetching {len(grant_ids)} specific grants by ID...[/yellow]")
            
            async def fetch_grants_by_ids():
                async with client:
                    grants = []
                    semaphore = asyncio.Semaphore(APIConfig.DEFAULT_MAX_CONCURRENT)
                    
                    async def fetch_single_grant(grant_id: str):
                        async with semaphore:
                            try:
                                return await client.get_grant(grant_id)
                            except RLAError as e:
                                logger.warning(f"Failed to fetch grant {grant_id}: {e}")
                                return None
                    
                    # Fetch all grants concurrently
                    tasks = [fetch_single_grant(grant_id) for grant_id in grant_ids]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Filter out None results and exceptions
                    for result in results:
                        if isinstance(result, Grant):
                            grants.append(result)
                        elif isinstance(result, Exception):
                            logger.warning(f"Exception fetching grant: {result}")
                    
                    return grants
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching grants by ID...", total=None)
                grants = run_async(fetch_grants_by_ids())
                progress.update(task, completed=100, description=f"Fetched {len(grants)} grants")
            
            # Create a mock response for consistency
            response = SearchResponse(
                total_results=len(grants),
                current_page=1,
                from_index=0,
                size=len(grants),
                results=grants
            )
        else:
            # Build filter query from individual parameters
            filter_query = build_filter_query(
                FilterConfig.GRANT_FILTERS,
                status=status,
                funder=funder,
                funding_scheme=funding_scheme,
                funding_amount=funding_amount,
                country=country,
                state=state,
                for_subject=for_subject,
                seo_subject=seo_subject
            )
            
            # Build advanced query for topic-based search
            advanced_query = build_advanced_query(topic=topic) if topic else ""
            
            # Handle --all option or limit-based fetching
            if all_results:
                console.print("[yellow]Fetching ALL grants from all pages...[/yellow]")
                
                async def fetch_all_async():
                    async with client:
                        response = await client.search_all_grants(
                            value=query,
                            filter_query=filter_query,
                            advanced_search_query=advanced_query,
                            max_concurrent=APIConfig.DEFAULT_MAX_CONCURRENT
                        )
                        return response
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Fetching all grants...", total=None)
                    response = run_async(fetch_all_async())
                    progress.update(task, completed=100, description=f"Fetched {len(response.results)} grants")
                
                grants = response.results
            elif limit > APIConfig.MAX_PAGE_SIZE:
                # For larger limits, use async bulk fetching
                console.print(f"[yellow]Fetching {limit} results using concurrent requests...[/yellow]")
                
                async def fetch_limited_async():
                    async with client:
                        # Use search_all but limit results
                        response = await client.search_all_grants(
                            value=query,
                            filter_query=filter_query,
                            advanced_search_query=advanced_query,
                            max_concurrent=APIConfig.DEFAULT_MAX_CONCURRENT
                        )
                        # Limit results to requested amount
                        response.results = response.results[:limit]
                        return response
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Fetching grants concurrently...", total=None)
                    response = run_async(fetch_limited_async())
                    progress.update(task, completed=100, description=f"Fetched {len(response.results)} grants")
                
                grants = response.results
            else:
                # Single page fetch
                page_size = min(limit, APIConfig.MAX_PAGE_SIZE)
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Searching grants...", total=None)
                    response = client.search_grants_sync(
                        value=query,
                        filter_query=filter_query,
                        page_number=1,
                        page_size=page_size,
                        advanced_search_query=advanced_query
                    )
                    progress.update(task, completed=100)
                
                grants = response.results
            
        if min_amount:
            grants = filter_grants_by_amount(grants, min_amount=min_amount)
        
        if show_stats and grants:
            stats = get_funding_statistics(grants)
            stats_panel = Panel(
                f"Total Funding: ${stats['total_funding']:,.2f}\n"
                f"Average Funding: ${stats['average_funding']:,.2f}\n"
                f"Median Funding: ${stats['median_funding']:,.2f}\n"
                f"Number of Grants: {stats['grant_count']}",
                title="Funding Statistics",
                border_style="green"
            )
            console.print(stats_panel)
        
        data = [asdict(g) for g in grants]
        
        if json_file:
            save_json_to_file(data, json_file)
        elif jsonl:
            output_jsonl_to_stdout(data)
        elif jsonl_file:
            save_jsonl_to_file(data, jsonl_file)
        else:
            # Display table format to stdout
            if all_results:
                info_panel = Panel(
                    f"Found {response.total_results} grants total\n"
                    f"Showing ALL {len(grants)} results",
                    title="Search Results",
                    border_style="blue"
                )
            else:
                info_panel = Panel(
                    f"Found {response.total_results} grants total\n"
                    f"Showing {len(grants)} results (limit: {limit})",
                    title="Search Results",
                    border_style="blue"
                )
            console.print(info_panel)
            
            if grants:
                table = format_grant_table(grants, f"Grants: {query}")
                console.print(table)
            else:
                console.print("[yellow]No grants found.[/yellow]")
                
    except RLAError as e:
        console.print(f"[red]API Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@grants_app.command("enrich")
def enrich_grants(
    input_file: str = typer.Argument(..., help="JSON file with grant records to enrich with grant_summary"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for enriched results (default: input_file_enriched.json)"),
    debug: bool = typer.Option(False, "--debug", help="Show full traceback on errors")
):
    """Enrich grant records from a JSON file with grant summaries from the RLA API"""
    global debug_mode
    debug_mode = debug
    try:
        client = get_client()
        
        console.print(f"[yellow]Processing input file: {input_file}[/yellow]")
        
        # Load JSON file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_records = json.load(f)
        except FileNotFoundError:
            console.print(f"[red]Error: Input file '{input_file}' not found[/red]")
            raise typer.Exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error: Invalid JSON in input file '{input_file}': {e}[/red]")
            raise typer.Exit(1)
        
        if not isinstance(input_records, list):
            console.print(f"[red]Error: Input file must contain a JSON array of records[/red]")
            raise typer.Exit(1)
        
        console.print(f"[yellow]Found {len(input_records)} records to process...[/yellow]")
        
        async def enrich_records_async():
            async with client:
                return await enrich_grants_with_summaries(client, input_records)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Enriching grant records with summaries...", total=None)
            enriched_records = run_async(enrich_records_async())
            progress.update(task, completed=100, description=f"Processed {len(enriched_records)} records")
        
        # Count successful enrichments
        successful_count = sum(1 for record in enriched_records if record.get('grant_summary', ''))
        console.print(f"[green]Successfully enriched {successful_count}/{len(enriched_records)} records with grant summaries[/green]")
        
        # Determine output file
        if not output_file:
            output_file = input_file.replace('.json', '_enriched.json')
            if output_file == input_file:  # If input doesn't end with .json
                output_file = input_file + '_enriched.json'
        
        # Save enriched records
        save_json_to_file(enriched_records, output_file)
        
    except RLAError as e:
        console.print(f"[red]API Error: {e}[/red]")
        if debug_mode:
            console.print("\n[yellow]Full traceback:[/yellow]")
            console.print(traceback.format_exc())
        raise typer.Exit(1)
    except Exception as e:
        handle_exception(e, "grant enrichment")
        raise typer.Exit(1)


@app.command("organisations")
def search_organisations(
    query: str = typer.Argument("", help="Search query for organisations (empty returns all organisations)"),
    limit: int = typer.Option(CLIConfig.DEFAULT_LIMIT, "--limit", "-l", help="Maximum number of results to return"),
    all_results: bool = typer.Option(False, "--all", help="Fetch all results from all pages (ignores --limit)"),
    country: Optional[str] = typer.Option(None, "--country", help="Filter by country"),
    state: Optional[str] = typer.Option(None, "--state", help="Filter by state/province"),
    type: Optional[str] = typer.Option(None, "--type", help="Filter by organisation type"),
    topic: Optional[str] = typer.Option(None, "--topic", help="Filter by research topic/field"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save results to JSON file"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Output JSON Lines to stdout"),
    jsonl_file: Optional[str] = typer.Option(None, "--jsonl-file", help="Save results to JSON Lines file"),
    debug: bool = typer.Option(False, "--debug", help="Show full traceback on errors")
):
    """Search for organisations in the RLA database"""
    global debug_mode
    debug_mode = debug
    
    options_provided = sum([json_file is not None, jsonl, jsonl_file is not None])
    if options_provided > 1:
        console.print("[red]Error: --json, --jsonl, and --jsonl-file are mutually exclusive[/red]")
        raise typer.Exit(1)
    
    try:
        client = get_client()
        
        # Build filter query using individual parameters
        filter_query = build_filter_query(
            FilterConfig.ORGANISATION_FILTERS,
            country=country,
            state=state,
            type=type,
            topic=topic,
            status=status
        )
        
        # Handle --all option or limit-based fetching
        if all_results:
            console.print("[yellow]Fetching ALL organisations from all pages...[/yellow]")
            
            async def search_all_async():
                async with client:
                    return await client.search_all_organisations(
                        value=query,
                        filter_query=filter_query,
                        max_concurrent=APIConfig.DEFAULT_MAX_CONCURRENT
                    )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching all organisations...", total=None)
                response = run_async(search_all_async())
                progress.update(task, completed=100, description=f"Found {len(response.results)} organisations")
            
            orgs = response.results
        else:
            # Use async search with limit and concurrent requests
            console.print(f"[yellow]Searching organisations with limit {limit}...[/yellow]")
            
            async def search_async():
                async with client:
                    # Calculate appropriate page size for efficiency
                    page_size = min(limit, APIConfig.MAX_PAGE_SIZE)
                    pages_needed = (limit + page_size - 1) // page_size
                    
                    if pages_needed == 1:
                        # Single page request
                        return await client.search_organisations(
                            value=query,
                            filter_query=filter_query,
                            page_number=1,
                            page_size=page_size
                        )
                    else:
                        # Multiple pages needed - fetch them concurrently
                        all_orgs = []
                        remaining = limit
                        
                        # Create semaphore to limit concurrent requests
                        semaphore = asyncio.Semaphore(APIConfig.DEFAULT_MAX_CONCURRENT)
                        
                        async def fetch_page(page_num: int, page_limit: int):
                            async with semaphore:
                                return await client.search_organisations(
                                    value=query,
                                    filter_query=filter_query,
                                    page_number=page_num,
                                    page_size=page_limit
                                )
                        
                        # Fetch pages concurrently
                        page_tasks = []
                        for page_num in range(1, pages_needed + 1):
                            current_page_size = min(remaining, page_size)
                            page_tasks.append(fetch_page(page_num, current_page_size))
                            remaining -= current_page_size
                            if remaining <= 0:
                                break
                        
                        page_results = await asyncio.gather(*page_tasks, return_exceptions=True)
                        
                        # Collect results from successful pages
                        total_results = 0
                        for result in page_results:
                            if not isinstance(result, Exception):
                                all_orgs.extend(result.results)
                                total_results = result.total_results  # Keep the last total
                            else:
                                logger.warning(f"Failed to fetch page: {result}")
                        
                        # Limit to requested amount
                        all_orgs = all_orgs[:limit]
                        
                        return SearchResponse(
                            total_results=total_results,
                            current_page=1,
                            from_index=0,
                            size=len(all_orgs),
                            results=all_orgs
                        )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Searching organisations concurrently...", total=None)
                response = run_async(search_async())
                progress.update(task, completed=100, description=f"Found {len(response.results)} organisations")
            
            orgs = response.results
        
        data = [asdict(org) for org in orgs]
        
        if json_file:
            save_json_to_file(data, json_file)
        elif jsonl:
            output_jsonl_to_stdout(data)
        elif jsonl_file:
            save_jsonl_to_file(data, jsonl_file)
        else:
            # Table format (default)
            if all_results:
                info_panel = Panel(
                    f"Found ALL {len(orgs)} organisations\n"
                    f"Total available: {response.total_results}\n"
                    f"Filters applied: {', '.join([f'{k}={v}' for k, v in [('country', country), ('state', state), ('type', type), ('topic', topic), ('status', status)] if v])}",
                    title="Search Results",
                    border_style="blue"
                )
            else:
                info_panel = Panel(
                    f"Found {len(orgs)} organisations (limited to {limit})\n"
                    f"Total available: {response.total_results}\n"
                    f"Filters applied: {', '.join([f'{k}={v}' for k, v in [('country', country), ('state', state), ('type', type), ('topic', topic), ('status', status)] if v])}",
                    title="Search Results",
                    border_style="blue"
                )
            console.print(info_panel)
            
            if orgs:
                table = format_organisation_table(orgs, f"Organisations: {query}")
                console.print(table)
            else:
                console.print("[yellow]No organisations found.[/yellow]")
                
    except RLAError as e:
        console.print(f"[red]API Error: {e}[/red]")
        if debug_mode:
            console.print("\n[yellow]Full traceback:[/yellow]")
            console.print(traceback.format_exc())
        raise typer.Exit(1)
    except Exception as e:
        handle_exception(e, "organisation search")
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI"""
    # Load environment variables from .env file in current working directory
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
