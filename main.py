import asyncio
import json
import logging
import time
import requests
import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional
from cachetools import TTLCache
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import uuid
import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:8001",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = TTLCache(maxsize=100, ttl=3600)
cookie_cache = TTLCache(maxsize=1, ttl=3600)

class Product(BaseModel):
    product_id: str
    title: str
    brand: str
    category: str
    final_price: float
    mrp: float
    discount_percentage: str
    rating_average: float
    rating_count: int
    product_url: str
    image_url: str
    source: str
    source_logo: str  # Added source_logo field
    availability: str
    color: Optional[str] = None
    fabric: Optional[str] = None
    pattern: Optional[str] = None
    occasion: Optional[str] = None
    type: Optional[str] = None
    kurta_length: Optional[str] = None
    sleeve_length: Optional[str] = None

    @field_validator("product_id", mode="before")
    def validate_product_id(cls, value):
        if value is None:
            return ""
        return str(value)

    @field_validator("discount_percentage", mode="before")
    def handle_discount(cls, value):
        if value is None or value == "":
            return "0"
        try:
            value = value.replace("(", "").replace(")", "").replace("% OFF", "").strip()
            return value if value else "0"
        except Exception as e:
            logger.warning(f"Invalid discount_percentage: {value}, defaulting to 0")
            return "0"

def get_myntra_cookies():
    cache_key = "myntra_cookies"
    if cache_key in cookie_cache:
        logger.info("Returning cached Myntra cookies")
        return cookie_cache[cache_key]

    logger.info("Fetching fresh Myntra cookies using Selenium...")
    start_time = time.time()
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Mobile Safari/537.36"
    )

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    try:
        driver.get("https://www.myntra.com/")
        WebDriverWait(driver, 5).until(lambda d: len(d.get_cookies()) > 0)
        cookies = driver.get_cookies()
        cookie_dict = {cookie["name"]: cookie["value"] for cookie in cookies}
        logger.info(f"Fetched {len(cookie_dict)} Myntra cookies in {time.time() - start_time:.2f} seconds")
        cookie_cache[cache_key] = cookie_dict
        return cookie_dict
    except Exception as e:
        logger.error(f"Error fetching Myntra cookies: {e}")
        return {}
    finally:
        driver.quit()

myntra_headers = {
    "accept": "*/*",
    "accept-language": "en-IN,en-US;q=0.9,en-GB;q=0.8,en;q=0.7,hi;q=0.6",
    "newrelic": "eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjMwNjIwNzEiLCJhcCI6IjcxODQwMDg2OSIsImlkIjoiYzliMTRkY2U3MjZhMzAxZiIsInRyIjoiMzFkZDhmMzA0YzAwZjI5YjNlMzFkMTY4MmFkMzQwYmYiLCJ0aSI6MTc0NzMwMDk3MDA2OCwidGsiOiI2Mjk1Mjg2In19",
    "pagination-context": '{"refresh":true, "v":1.0}',
    "priority": "u=1, i",
    "referer": "https://www.myntra.com/",
    "sec-ch-ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
    "sec-ch-ua-mobile": "?1",
    "sec-ch-ua-platform": '"Android"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "traceparent": "00-31dd8f304c00f29b3e31d1682ad340bf-c9b14dce726a301f-01",
    "tracestate": "6295286@nr=0-1-3062071-718400869-c9b14dce726a301f----1747300970068",
    "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Mobile Safari/537.36",
    "x-device-state": "model=Nexus 5;brand=OnePlus",
    "x-location-context": "pincode=362020;source=USER",
    "x-meta-app": "appFamily=MyntraRetailMweb;",
    "x-myntra-abtest": "config.bucket=regular;pdp.desktop.savedAddress=enabled;coupon.cart.channelAware=channelAware_Enabled;returns.obd=enabled;payments.twid=enabled;payments.simplpayin3=enabled;payments.tokenization.autocheck=enabled;cart.cartfiller.personalised=enabled;payment.default.reco.nu=enabled",
    "x-myntra-app": f"deviceID={str(uuid.uuid4())};reqChannel=web;appFamily=MyntraRetailMweb;",
    "x-myntraweb": "Yes",
    "x-requested-with": "browser",
}

def extract_attributes_myntra(product):
    title = product.get("productName", "").lower()
    color = None
    fabric = None
    pattern = None
    occasion = None
    type = None
    kurta_length = None
    sleeve_length = None

    colors = ["blue", "red", "green", "purple", "yellow", "black", "white"]
    for c in colors:
        if c in title:
            color = c.capitalize()
            break

    fabrics = ["cotton", "viscose rayon", "khadi"]
    for f in fabrics:
        if f in title:
            fabric = f.capitalize()
            break

    patterns = ["printed", "solid", "embroidered"]
    for p in patterns:
        if p in title:
            pattern = p.capitalize()
            break

    occasions = ["casual", "festive"]
    for o in occasions:
        if o in title:
            occasion = o.capitalize()
            break

    types = ["a-line", "straight"]
    for t in types:
        if t in title:
            type = t.replace("-", " ").capitalize()
            break

    lengths = ["knee length", "calf length"]
    for l in lengths:
        if l in title:
            kurta_length = l.capitalize()
            break

    sleeves = ["three quarter", "short"]
    for s in sleeves:
        if s in title:
            sleeve_length = s.capitalize()
            break

    return {
        "color": color,
        "fabric": fabric,
        "pattern": pattern,
        "occasion": occasion,
        "type": type,
        "kurta_length": kurta_length,
        "sleeve_length": sleeve_length
    }

def fetch_myntra_data(query, cookies):
    cache_key = f"myntra_{query}"
    if cache_key in cache:
        logger.info(f"Returning cached Myntra data for {cache_key}")
        return cache[cache_key], None

    start_time = time.time()
    url = f"https://www.myntra.com/gateway/v2/search/{query.replace(' ', '-')}"
    params = {"o": 0, "ifo": 0, "ifc": 0, "pincode": "362020", "rows": 100, "priceBuckets": 20}
    try:
        response = requests.get(url, params=params, headers=myntra_headers, cookies=cookies)
        logger.info(
            f"Myntra search for '{query}' - Status: {response.status_code}, Time: {time.time() - start_time:.2f} seconds")
        response.raise_for_status()
        data = response.json()
        products = data.get("products", [])
        extracted_data = []
        for product in products:
            try:
                attributes = extract_attributes_myntra(product)
                product_info = {
                    "product_id": str(product.get("productId", "")),
                    "title": product.get("productName", ""),
                    "brand": product.get("brand", ""),
                    "category": product.get("category", ""),
                    "final_price": float(product.get("price", 0)),
                    "mrp": float(product.get("mrp", 0)),
                    "discount_percentage": product.get("discountDisplayLabel", ""),
                    "rating_average": float(product.get("rating", 0)),
                    "rating_count": int(product.get("ratingCount", 0)),
                    "product_url": f"https://www.myntra.com/{product.get('landingPageUrl', '')}",
                    "image_url": product.get("images", [{}])[0].get("src", ""),
                    "source": "Myntra",
                    "source_logo": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSUrnmr3CB1oDs0WqiWPzNxENXCnRE-1yKVKw&s",
                    "availability": "In Stock" if product.get("inventoryInfo", [{}])[0].get("available", False) else "Out of Stock",
                    "color": attributes["color"],
                    "fabric": attributes["fabric"],
                    "pattern": attributes["pattern"],
                    "occasion": attributes["occasion"],
                    "type": attributes["type"],
                    "kurta_length": attributes["kurta_length"],
                    "sleeve_length": attributes["sleeve_length"]
                }
                extracted_data.append(product_info)
            except Exception as e:
                logger.error(f"Error extracting Myntra product: {e}")
        cache[cache_key] = extracted_data
        logger.info(f"Extracted {len(extracted_data)} Myntra products in {time.time() - start_time:.2f} seconds")
        return extracted_data, None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Myntra data for '{query}': {e}")
        return [], f"Failed to fetch Myntra data for '{query}'."

def get_flipkart_cookies(query):
    cache_key = "flipkart_cookies"
    if cache_key in cookie_cache:
        logger.info("Returning cached Flipkart cookies")
        return cookie_cache[cache_key]

    logger.info("Fetching fresh Flipkart cookies using Selenium...")
    start_time = time.time()
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    try:
        driver.get(f"https://www.flipkart.com/search?q={query}")
        WebDriverWait(driver, 5).until(lambda d: len(d.get_cookies()) > 0)
        cookies = driver.get_cookies()
        cookie_dict = {cookie["name"]: cookie["value"] for cookie in cookies}
        logger.info(f"Fetched {len(cookie_dict)} Flipkart cookies in {time.time() - start_time:.2f} seconds")
        cookie_cache[cache_key] = cookie_dict
        return cookie_dict
    except Exception as e:
        logger.error(f"Error fetching Flipkart cookies: {e}")
        return {}
    finally:
        driver.quit()

async def fetch_flipkart_data(query, page, session, max_retries=3):
    cache_key = f"flipkart_{query}_page_{page}"
    if cache_key in cache:
        logger.info(f"Returning cached Flipkart data for {cache_key}")
        return cache[cache_key]

    start_time = time.time()
    url = "https://2.rome.api.flipkart.com/api/4/page/fetch"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/json",
        "Origin": "https://www.flipkart.com",
        "Referer": f"https://www.flipkart.com/search?q={query}",
        "X-User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 FKUA/website/42/website/Desktop",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    }
    payload = {
        "pageUri": f"/search?q={query}&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&as-pos=1&as-type=HISTORY&page={page}",
        "pageContext": {
            "fetchSeoData": True,
            "paginatedFetch": True,
            "pageNumber": page,
            "paginationContextMap": {
                "federator": {
                    "pageNumber": page,
                    "PRODUCT": 40,
                    "marketplaceProductsOffset": 40 * (page - 1),
                    "productsOffset": 40 * (page - 1),
                }
            },
        },
        "requestContext": {"type": "BROWSE_PAGE", "ssid": "test_ssid", "sqid": "test_sqid"},
    }

    cookies = get_flipkart_cookies(query)
    for attempt in range(1, max_retries + 1):
        try:
            async with session.post(url, headers=headers, json=payload, cookies=cookies, timeout=10) as response:
                logger.info(
                    f"Flipkart attempt {attempt} for page {page} - Status: {response.status}, Time: {time.time() - start_time:.2f} seconds")
                if response.status != 200:
                    if attempt == max_retries:
                        logger.warning("Mocking Flipkart data due to repeated failures")
                        return {
                            "RESPONSE": {
                                "slots": [{
                                    "slotType": "WIDGET",
                                    "widget": {
                                        "type": "PRODUCT_SUMMARY",
                                        "data": {
                                            "products": [{
                                                "productInfo": {
                                                    "value": {
                                                        "id": f"mock_{page}_{attempt}",
                                                        "titles": {"title": "Mock Jeans"},
                                                        "productBrand": "Mock Brand",
                                                        "analyticsData": {"category": "Jeans"},
                                                        "pricing": {
                                                            "finalPrice": {"decimalValue": 999.0},
                                                            "mrp": {"decimalValue": 1999.0},
                                                            "totalDiscount": 50
                                                        },
                                                        "rating": {"average": 4.5, "count": 100},
                                                        "smartUrl": "https://www.flipkart.com/mock-jeans",
                                                        "media": {"images": [{"url": "https://via.placeholder.com/150"}]}
                                                    }
                                                }
                                            }]
                                        }
                                    }
                                }]
                            }
                        }
                    await asyncio.sleep(2)
                    continue
                data = await response.json()
                cache[cache_key] = data
                return data
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Flipkart request failed: {e}")
            if attempt == max_retries:
                logger.warning("Mocking Flipkart data due to repeated failures")
                return {
                    "RESPONSE": {
                        "slots": [{
                            "slotType": "WIDGET",
                            "widget": {
                                "type": "PRODUCT_SUMMARY",
                                "data": {
                                    "products": [{
                                        "productInfo": {
                                            "value": {
                                                "id": f"mock_{page}_{attempt}",
                                                "titles": {"title": "Mock Jeans"},
                                                "productBrand": "Mock Brand",
                                                "analyticsData": {"category": "Jeans"},
                                                "pricing": {
                                                    "finalPrice": {"decimalValue": 999.0},
                                                    "mrp": {"decimalValue": 1999.0},
                                                    "totalDiscount": 50
                                                },
                                                "rating": {"average": 4.5, "count": 100},
                                                "smartUrl": "https://www.flipkart.com/mock-jeans",
                                                "media": {"images": [{"url": "https://via.placeholder.com/150"}]},
                                                "source_logo": "https://static.vecteezy.com/system/resources/previews/054/650/802/non_2x/flipkart-logo-rounded-flipkart-logo-free-download-flipkart-logo-free-png.png"
                                            }
                                        }
                                    }]
                                }
                            }
                        }]
                    }
                }
            await asyncio.sleep(2)
    return None

def extract_attributes_flipkart(product_info):
    title = product_info.get("titles", {}).get("title", "").lower()
    color = None
    fabric = None
    pattern = None
    occasion = None
    type = None
    kurta_length = None
    sleeve_length = None

    colors = ["blue", "red", "green", "purple", "yellow", "black", "white"]
    for c in colors:
        if c in title:
            color = c.capitalize()
            break

    fabrics = ["cotton", "viscose rayon", "khadi"]
    for f in fabrics:
        if f in title:
            fabric = f.capitalize()
            break

    patterns = ["printed", "solid", "embroidered"]
    for p in patterns:
        if p in title:
            pattern = p.capitalize()
            break

    occasions = ["casual", "festive"]
    for o in occasions:
        if o in title:
            occasion = o.capitalize()
            break

    types = ["a-line", "straight"]
    for t in types:
        if t in title:
            type = t.replace("-", " ").capitalize()
            break

    lengths = ["knee length", "calf length"]
    for l in lengths:
        if l in title:
            kurta_length = l.capitalize()
            break

    sleeves = ["three quarter", "short"]
    for s in sleeves:
        if s in title:
            sleeve_length = s.capitalize()
            break

    return {
        "color": color,
        "fabric": fabric,
        "pattern": pattern,
        "occasion": occasion,
        "type": type,
        "kurta_length": kurta_length,
        "sleeve_length": sleeve_length
    }

def parse_flipkart_products(data, query, page):
    start_time = time.time()
    products = []
    if not data or "RESPONSE" not in data:
        logger.warning("No 'RESPONSE' key in Flipkart JSON")
        return products

    slots = data["RESPONSE"].get("slots", [])
    for slot in slots:
        if slot.get("slotType") == "WIDGET" and slot.get("widget", {}).get("type") == "PRODUCT_SUMMARY":
            slot_products = slot.get("widget", {}).get("data", {}).get("products", [])
            for product in slot_products:
                product_info = product.get("productInfo", {}).get("value", {})
                pricing = product_info.get("pricing", {})
                image_url = (
                    product_info.get("media", {}).get("images", [{}])[0].get("url", "") or
                    product_info.get("media", {}).get("imageList", [{}])[0].get("url", "") or
                    product_info.get("media", {}).get("mediaImages", [{}])[0].get("url", "")
                )
                if image_url:
                    image_url = image_url.replace("{@width}", "400").replace("{@height}", "400").replace("{@quality}", "70")
                attributes = extract_attributes_flipkart(product_info)
                product_data = {
                    "product_id": str(product_info.get("id", "")),
                    "title": product_info.get("titles", {}).get("title", ""),
                    "brand": product_info.get("productBrand", ""),
                    "category": product_info.get("analyticsData", {}).get("category", ""),
                    "final_price": float(pricing.get("finalPrice", {}).get("decimalValue", 0.0)),
                    "mrp": float(pricing.get("mrp", {}).get("decimalValue", 0.0)),
                    "discount_percentage": str(pricing.get("totalDiscount", 0)),
                    "rating_average": float(product_info.get("rating", {}).get("average", 0.0)),
                    "rating_count": int(product_info.get("rating", {}).get("count", 0)),
                    "product_url": product_info.get("smartUrl", ""),
                    "image_url": image_url,
                    "source": "Flipkart",
                    "source_logo": "https://static.vecteezy.com/system/resources/previews/054/650/802/non_2x/flipkart-logo-rounded-flipkart-logo-free-download-flipkart-logo-free-png.png",
                    "availability": product_info.get("availability", {}).get("displayState", "In Stock"),
                    "color": attributes["color"],
                    "fabric": attributes["fabric"],
                    "pattern": attributes["pattern"],
                    "occasion": attributes["occasion"],
                    "type": attributes["type"],
                    "kurta_length": attributes["kurta_length"],
                    "sleeve_length": attributes["sleeve_length"]
                }
                products.append(product_data)
    logger.info(
        f"Extracted {len(products)} Flipkart products from page {page} in {time.time() - start_time:.2f} seconds")
    return products

async def fetch_and_filter_products(
        query: str,
        page: int = 1,
        price_range: Optional[str] = None,
        colors: Optional[str] = None,
        brands: Optional[str] = None,
        sort: Optional[str] = None,
        for_api: bool = False
):
    start_time = time.time()
    query = query.strip() or "jeans"

    myntra_cookies = get_myntra_cookies()
    myntra_products, myntra_error = fetch_myntra_data(query, myntra_cookies)

    async with aiohttp.ClientSession() as session:
        flipkart_tasks = [fetch_flipkart_data(query, page_num, session) for page_num in range(1, 6)]
        flipkart_results = await asyncio.gather(*flipkart_tasks, return_exceptions=True)
        flipkart_products = []
        for page_num, data in enumerate(flipkart_results, 1):
            if isinstance(data, dict) and data:
                flipkart_products.extend(parse_flipkart_products(data, query, page_num))

    unique_myntra = []
    seen_myntra_ids = set()
    for product in myntra_products:
        if product["product_id"] not in seen_myntra_ids:
            seen_myntra_ids.add(product["product_id"])
            try:
                unique_myntra.append(Product(**product).model_dump() if not for_api else Product(**product))
            except Exception as e:
                logger.error(f"Validation error for Myntra product: {product}, Error: {e}")
                continue

    unique_flipkart = []
    seen_flipkart_ids = set()
    for product in flipkart_products:
        if product["product_id"] not in seen_flipkart_ids:
            seen_flipkart_ids.add(product["product_id"])
            try:
                unique_flipkart.append(Product(**product).model_dump() if not for_api else Product(**product))
            except Exception as e:
                logger.error(f"Validation error for Flipkart product: {product}, Error: {e}")
                continue

    all_products = unique_myntra + unique_flipkart

    # Extract unique colors and brands
    unique_colors = sorted(set(p["color"] for p in all_products if p["color"]))
    unique_brands = sorted(set(p["brand"] for p in all_products if p["brand"]))

    # Apply filters
    filtered_products = all_products

    # Price range filter
    price_min = 0
    price_max = float('inf')
    if price_range:
        try:
            if price_range == "5000+":
                price_min = 5000
                price_max = float('inf')
            else:
                min_str, max_str = price_range.split('-')
                price_min = float(min_str)
                price_max = float(max_str)
        except ValueError:
            logger.warning(f"Invalid price_range: {price_range}, defaulting to all prices")
            price_range = ""

    filtered_products = [
        p for p in filtered_products
        if p["final_price"] >= price_min and p["final_price"] <= price_max
    ]

    # Color filter
    if colors:
        color_list = colors.split(",") if isinstance(colors, str) else colors
        filtered_products = [
            p for p in filtered_products
            if p["color"] and p["color"] in color_list
        ]

    # Brand filter
    if brands:
        brand_list = brands.split(",") if isinstance(brands, str) else brands
        filtered_products = [
            p for p in filtered_products
            if p["brand"] and p["brand"] in brand_list
        ]

    # Apply sorting
    if sort:
        if sort == "price-asc":
            filtered_products.sort(key=lambda x: x["final_price"] if not for_api else x.final_price)
        elif sort == "price-desc":
            filtered_products.sort(key=lambda x: x["final_price"] if not for_api else x.final_price, reverse=True)
        elif sort == "discount":
            filtered_products.sort(key=lambda x: int(x["discount_percentage"] if not for_api else x.discount_percentage), reverse=True)

    # Pagination logic
    products_per_page = 20
    total_products = len(filtered_products)
    total_pages = max(1, (total_products + products_per_page - 1) // products_per_page)
    page = max(1, min(page, total_pages))
    start_idx = (page - 1) * products_per_page
    end_idx = start_idx + products_per_page
    paginated_products = filtered_products[start_idx:end_idx]

    logger.info(
        f"Returning {len(paginated_products)} products for query '{query}' (Page {page}/{total_pages}, Total: {total_products}, Myntra: {len(unique_myntra)}, Flipkart: {len(unique_flipkart)}) in {time.time() - start_time:.2f} seconds")

    return {
        "products": paginated_products,
        "error": myntra_error,
        "current_page": page,
        "total_pages": total_pages,
        "query": query,
        "price_range": price_range or "",
        "colors": unique_colors,
        "brands": unique_brands,
        "sort": sort or "price-asc"
    }

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "products": [],
            "error": None,
            "current_page": 1,
            "total_pages": 1,
            "query": "",
            "price_range": "",
            "colors": [],
            "brands": [],
            "sort": "price-asc"
        }
    )

@app.get("/search", response_class=HTMLResponse)
async def search_get(
        request: Request,
        query: str = "",
        page: int = 1,
        price_range: Optional[str] = None,
        colors: Optional[str] = None,
        brands: Optional[str] = None,
        sort: Optional[str] = None
):
    result = await fetch_and_filter_products(query, page, price_range, colors, brands, sort)
    return templates.TemplateResponse("index.html", {"request": request, **result})

@app.post("/search", response_class=HTMLResponse)
async def search_post(request: Request):
    form = await request.form()
    query = form.get("query", "").strip()
    page = int(form.get("page", 1))
    price_range = form.get("price_range", "")
    colors = form.get("colors", "")
    brands = form.get("brands", "")
    sort = form.get("sort", "price-asc")
    result = await fetch_and_filter_products(query, page, price_range, colors, brands, sort)
    return templates.TemplateResponse("index.html", {"request": request, **result})

@app.get("/api/products", response_model=List[Product])
async def api_products(
        query: str = "jeans",
        page: int = 1,
        price_range: Optional[str] = None,
        colors: Optional[str] = None,
        brands: Optional[str] = None,
        sort: Optional[str] = None
):
    result = await fetch_and_filter_products(query, page, price_range, colors, brands, sort, for_api=True)
    return result["products"]

@app.get("/proxy-image")
async def proxy_image(url: str):
    url = url.replace("{@width}", "400").replace("{@height}", "400").replace("{@quality}", "70")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Referer": "https://www.flipkart.com/",
            "Accept": "image/*"
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return Response(content=response.content, media_type=response.headers.get("Content-Type", "image/jpeg"))
    except Exception as e:
        logger.error(f"Failed to proxy image {url}: {e}")
        return Response(content=requests.get("https://via.placeholder.com/150").content, media_type="image/jpeg")

@app.get("/ping")
async def ping():
    logger.info("Ping request received")
    return {"message": "pong"}

if platform.system() == "Emscripten":
    asyncio.ensure_future(app.router.startup())
else:
    import uvicorn

    if __name__ == "__main__":
        logger.info("Server starting. Access the application at http://localhost:8001")
        uvicorn.run(app, host="0.0.0.0", port=8001)