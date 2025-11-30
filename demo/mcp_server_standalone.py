"""
FastMCP 기반 stdio MCP 목업 서버

모델이 tool_call로만 MODIS 이미지를 생성/조회하도록 한다.

- available: 가용성 확인
- list_modis_images: assets 내 PNG 목록
- get_modis_image: 파일을 base64/data URI로 반환
- fetch_modis_data: MODIS 이미지 생성 (모델이 호출할 때만)
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional, Tuple

from fastmcp import FastMCP

import get_modis
from get_modis import check_data_availability, initialize_earth_engine

mcp = FastMCP("MODIS FastMCP Server")

ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


@mcp.tool()
def available(
    date: str,
    hour: int,
    west: float = -125.0,
    south: float = 24.0,
    east: float = -66.0,
    north: float = 49.0,
) -> dict:
    try:
        bbox = (west, south, east, north)
        result = check_data_availability(date, hour, bbox)
        return {"success": True, **result}
    except Exception as exc:  # pragma: no cover
        return {"success": False, "error": str(exc)}


@mcp.tool()
def list_modis_images() -> dict:
    """
    [CRITICAL STEP 1] Retrieve a list of already generated/cached MODIS images in the local assets.
    
    ALWAYS call this tool FIRST before attempting to fetch new data.
    If a suitable image exists in this list, use 'get_modis_image' to retrieve it.
    Only proceed to 'fetch_modis_data' if the list is empty or does not contain the requested date/location.
    """
    try:
        files = sorted(f.name for f in ASSETS_DIR.glob("*.png"))
        return {"success": True, "files": files, "count": len(files)}
    except Exception as exc:  # pragma: no cover
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_modis_image(filename: str) -> dict:
    """
    [STEP 2-A] Retrieve the actual image data (Base64) for a specific filename found in 'list_modis_images'.
    Use this to display the image to the user without regenerating it.
    """
    try:
        image_path = ASSETS_DIR / filename
        if not image_path.exists():
            return {
                "success": False,
                "error": f"Image not found: {filename}",
                "available": sorted(f.name for f in ASSETS_DIR.glob('*.png')),
            }
        with image_path.open("rb") as fp:
            encoded = base64.b64encode(fp.read()).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded}"
        return {
            "success": True,
            "filename": filename,
            "image_base64": encoded,
            "data_uri": data_uri,
            "size_kb": round(len(encoded) / 1024, 2),
        }
    except Exception as exc:  # pragma: no cover
        return {"success": False, "error": str(exc)}


@mcp.tool()
def fetch_modis_data(
    date_str: str,
    satellite: str,
    west: float,
    south: float,
    east: float,
    north: float,
    is_daytime: bool = True,
    pinpoint_lat: Optional[float] = None,
    pinpoint_lng: Optional[float] = None,
) -> dict:
    """
    [STEP 2-B] Fetch and generate NEW MODIS satellite imagery from Earth Engine.
    
    WARNING: This operation is SLOW and computationally expensive.
    DO NOT use this tool if the image is already available in 'list_modis_images'.
    ONLY use this tool if you have confirmed the image does not exist locally.
    
    Returns the generated image data directly.
    """
    try:
        bbox: Tuple[float, float, float, float] = (west, south, east, north)
        pinpoint = (pinpoint_lat, pinpoint_lng) if pinpoint_lat is not None and pinpoint_lng is not None else None

        result_path = get_modis.fetch_modis_images(
            date_str=date_str,
            satellite=satellite,
            bbox=bbox,
            is_daytime=is_daytime,
            output_dir=str(ASSETS_DIR),
            image_size=(512, 512),
            pinpoint=pinpoint,
        )

        if not result_path:
            return {"success": False, "error": "Failed to fetch MODIS data", "date": date_str, "satellite": satellite}

        filename = Path(result_path).name
        file_size = Path(result_path).stat().st_size
        if file_size == 0:
            return {"success": False, "error": "Generated image is empty (0 bytes)", "filename": filename}

        with open(result_path, "rb") as fp:
            encoded = base64.b64encode(fp.read()).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded}"

        return {
            "success": True,
            "filename": filename,
            "image_base64": encoded,
            "data_uri": data_uri,
            "size_kb": round(len(encoded) / 1024, 2),
            "message": f"Successfully fetched MODIS data for {date_str} ({satellite})",
        }
    except Exception as exc:  # pragma: no cover
        return {"success": False, "error": str(exc)}


if __name__ == "__main__":
    print("🌍 Initializing Earth Engine...")
    initialize_earth_engine()
    print("🚀 Starting FastMCP stdio server")
    print(f"📁 Assets directory: {ASSETS_DIR}")
    print("🛠️  Tools: available, list_modis_images, get_modis_image, fetch_modis_data")
    mcp.run()
